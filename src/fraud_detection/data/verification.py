"""Verify the prepared splits before any model is trained.

Phase 1 wrote the splits to Parquet. Before training on them we re-check, from
the files on disk, that:

* each split still has the expected schema and valid values,
* row counts and fraud counts match what Phase 1 recorded (detects stale or
  edited files),
* there is no leakage: no shared rows, strict time order, both classes present,
* the Phase 1 preprocessor artifact and the code that rebuilds it agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data.loader import load_parquet
from fraud_detection.data.split import DataSplits
from fraud_detection.data.validation import ValidationReport, validate_raw_data, validate_splits
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "validation", "test")
PREP_REPORT_FILE = "data_preparation_report.json"


@dataclass
class VerifiedData:
    splits: DataSplits
    report: ValidationReport
    summary: dict[str, dict]


def load_splits(processed_dir: Path) -> DataSplits:
    """Load the train / validation / test Parquet files written by ``prepare``."""
    frames = {name: load_parquet(Path(processed_dir) / f"{name}.parquet") for name in SPLIT_NAMES}
    return DataSplits(**frames)


def _load_prep_report(artifacts_dir: Path) -> dict | None:
    path = Path(artifacts_dir) / PREP_REPORT_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def verify_prepared_data(cfg: Config) -> VerifiedData:
    """Load the splits and run all pre-training checks. Raises on failure."""
    target, time_col = cfg.dataset.target, cfg.dataset.time_column
    splits = load_splits(cfg.paths.processed_dir)

    # Leakage + usability checks across splits (reuses the Phase 1 validator).
    report = validate_splits(
        splits.as_dict(),
        strategy=cfg.split.strategy,
        min_positives=cfg.split.min_positives_per_split,
        target=target,
        time_column=time_col,
    )
    report.name = "prepared_data"

    # Schema / value checks on every split.
    for name, df in splits.as_dict().items():
        split_report = validate_raw_data(df)
        report.checks_run += split_report.checks_run
        report.errors += [f"[{name}] {e}" for e in split_report.errors]
        report.check(not df.duplicated().any(), f"[{name}] contains duplicate rows", fatal=True)

    # Consistency with what Phase 1 recorded.
    summary = splits.summary(target, time_col)
    prep = _load_prep_report(cfg.paths.artifacts_dir)
    if report.check(prep is not None, f"{PREP_REPORT_FILE} not found", fatal=False):
        for name in SPLIT_NAMES:
            expected = prep["splits"][name]
            for key in ("rows", "fraud_cases"):
                report.check(
                    summary[name][key] == expected[key],
                    f"[{name}] {key}={summary[name][key]} but Phase 1 recorded "
                    f"{expected[key]} - re-run `fraud-detection prepare`",
                )

    report.log()
    report.raise_if_failed()
    for name, stats in summary.items():
        logger.info(
            "Verified %-10s rows=%8s fraud=%4d (%.4f%%)",
            name,
            f"{stats['rows']:,}",
            stats["fraud_cases"],
            100 * stats["fraud_rate"],
        )
    return VerifiedData(splits=splits, report=report, summary=summary)


def check_preprocessor_consistency(
    fitted_preprocessor, artifacts_dir: Path, sample: pd.DataFrame
) -> bool | None:
    """Compare a freshly fitted preprocessor against the Phase 1 artifact.

    Both are fitted on the same training split with the same code, so their
    outputs must match. Returns ``None`` if the artifact does not exist.
    """
    from fraud_detection.pipelines.prepare import load_preprocessor

    try:
        saved = load_preprocessor(artifacts_dir)
    except FileNotFoundError:
        return None
    a = fitted_preprocessor.transform(sample)
    b = saved.transform(sample)
    return list(a.columns) == list(b.columns) and bool(((a - b).abs() < 1e-9).all().all())
