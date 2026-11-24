"""Data validation: catch bad data *before* it reaches a model.

Two levels of checks are provided:

* :func:`validate_raw_data` - is the raw table what we expect (schema, types,
  missing values, label values, value ranges)?
* :func:`validate_splits` - are the train / validation / test splits safe to
  use (no overlap, correct time order, enough fraud cases in each)?

Each check adds either an *error* (pipeline must stop) or a *warning* (worth
knowing, but not fatal) to a :class:`ValidationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraud_detection.data.schema import (
    AMOUNT_COLUMN,
    EXPECTED_COLUMNS,
    MAX_EXPECTED_TIME_SECONDS,
    TARGET_COLUMN,
    TIME_COLUMN,
    VALID_TARGET_VALUES,
)
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when a dataset fails one or more blocking validation checks."""


@dataclass
class ValidationReport:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def passed(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str, *, fatal: bool = True) -> bool:
        """Record one check. Returns ``condition`` so callers can branch on it."""
        self.checks_run += 1
        if not condition:
            (self.errors if fatal else self.warnings).append(message)
        return condition

    def log(self) -> None:
        status = "PASSED" if self.passed else "FAILED"
        logger.info(
            "Validation '%s' %s (%d checks, %d errors, %d warnings)",
            self.name,
            status,
            self.checks_run,
            len(self.errors),
            len(self.warnings),
        )
        for msg in self.warnings:
            logger.warning("[%s] %s", self.name, msg)
        for msg in self.errors:
            logger.error("[%s] %s", self.name, msg)

    def raise_if_failed(self) -> None:
        if not self.passed:
            details = "\n  - ".join(self.errors)
            raise DataValidationError(f"Validation '{self.name}' failed:\n  - {details}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "checks_run": self.checks_run,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_raw_data(df: pd.DataFrame, expected_rows: int | None = None) -> ValidationReport:
    """Validate the raw Kaggle table against the expected schema and value rules."""
    report = ValidationReport(name="raw_data")

    # --- Schema --------------------------------------------------------------
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if not report.check(not missing, f"Missing required columns: {missing}"):
        return report  # later checks depend on these columns existing
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    report.check(not extra, f"Unexpected extra columns: {extra}", fatal=False)
    report.check(len(df) > 0, "Dataset is empty")

    non_numeric = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if not report.check(not non_numeric, f"Non-numeric columns: {non_numeric}"):
        return report

    # --- Completeness --------------------------------------------------------
    n_missing = df[EXPECTED_COLUMNS].isna().sum()
    n_missing = n_missing[n_missing > 0]
    report.check(n_missing.empty, f"Missing values found: {n_missing.to_dict()}")

    values = df[EXPECTED_COLUMNS].to_numpy(dtype="float64")
    report.check(not np.isinf(values).any(), "Infinite values found")

    # --- Target --------------------------------------------------------------
    labels = set(df[TARGET_COLUMN].dropna().unique().tolist())
    report.check(
        labels <= VALID_TARGET_VALUES,
        f"Target '{TARGET_COLUMN}' has invalid values: {sorted(labels - VALID_TARGET_VALUES)}",
    )
    report.check(
        labels == VALID_TARGET_VALUES,
        f"Target must contain both classes 0 and 1, found {sorted(labels)}",
    )

    # --- Value ranges --------------------------------------------------------
    report.check(
        bool((df[AMOUNT_COLUMN].dropna() >= 0).all()), f"'{AMOUNT_COLUMN}' has negative values"
    )
    report.check(
        bool((df[TIME_COLUMN].dropna() >= 0).all()), f"'{TIME_COLUMN}' has negative values"
    )
    report.check(
        bool(df[TIME_COLUMN].max() <= MAX_EXPECTED_TIME_SECONDS),
        f"'{TIME_COLUMN}' exceeds the expected 2-day window ({MAX_EXPECTED_TIME_SECONDS}s)",
        fatal=False,
    )

    # --- Informational -------------------------------------------------------
    if expected_rows is not None:
        report.check(
            len(df) == expected_rows,
            f"Row count {len(df):,} differs from the published {expected_rows:,} "
            "(different file version?)",
            fatal=False,
        )
    n_dupes = int(df.duplicated().sum())
    report.check(
        n_dupes == 0,
        f"{n_dupes:,} exact duplicate rows found (removed during cleaning)",
        fatal=False,
    )
    return report


def _row_hashes(df: pd.DataFrame) -> set[int]:
    return set(pd.util.hash_pandas_object(df, index=False).tolist())


def validate_splits(
    splits: dict[str, pd.DataFrame],
    *,
    strategy: str,
    min_positives: int,
    target: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> ValidationReport:
    """Validate train/validation/test splits for leakage and usability.

    Args:
        splits: Ordered mapping, e.g. ``{"train": ..., "validation": ..., "test": ...}``.
        strategy: ``"time"`` additionally enforces strict chronological order.
        min_positives: Minimum number of fraud cases each split must contain.
    """
    report = ValidationReport(name="splits")
    names = list(splits)

    columns = [tuple(df.columns) for df in splits.values()]
    report.check(len(set(columns)) == 1, "Splits do not share identical columns")

    for name, df in splits.items():
        report.check(len(df) > 0, f"Split '{name}' is empty")
        positives = int(df[target].sum()) if len(df) else 0
        report.check(
            positives >= min_positives,
            f"Split '{name}' has only {positives} fraud cases (minimum {min_positives})",
        )
        if len(df):
            report.check(df[target].nunique() == 2, f"Split '{name}' does not contain both classes")

    # No identical transaction may appear in more than one split.
    hashes = {name: _row_hashes(df) for name, df in splits.items()}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = len(hashes[a] & hashes[b])
            report.check(overlap == 0, f"{overlap} identical rows appear in both '{a}' and '{b}'")

    # For a time split, every row of an earlier split must precede the next split.
    if strategy == "time":
        for earlier, later in zip(names, names[1:], strict=False):
            if len(splits[earlier]) and len(splits[later]):
                end = splits[earlier][time_column].max()
                start = splits[later][time_column].min()
                report.check(
                    end < start,
                    f"Time leakage: '{earlier}' ends at {end} but '{later}' starts at {start}",
                )
    return report
