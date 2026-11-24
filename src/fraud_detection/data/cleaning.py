"""Deterministic cleaning steps applied to the whole dataset before splitting.

Only *row-level, rule-based* operations belong here (nothing that learns from
the data, like computing a mean). That is what makes it safe to run before the
train/validation/test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from fraud_detection.data.schema import TARGET_COLUMN, TIME_COLUMN
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CleaningSummary:
    rows_before: int
    rows_after: int
    duplicates_removed: int
    fraud_duplicates_removed: int

    def to_dict(self) -> dict:
        return asdict(self)


def clean_data(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    target: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Remove exact duplicate rows and sort chronologically.

    Why remove duplicates before splitting? If the same transaction ends up in
    both train and test, the model is evaluated on rows it has memorised, which
    inflates test scores (a form of data leakage).
    """
    rows_before = len(df)
    fraud_dupes = 0
    if drop_duplicates:
        dupe_mask = df.duplicated(keep="first")
        fraud_dupes = int(df.loc[dupe_mask, target].sum())
        df = df.loc[~dupe_mask]

    # Stable sort keeps the original order for rows that share a timestamp.
    df = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)

    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=len(df),
        duplicates_removed=rows_before - len(df),
        fraud_duplicates_removed=fraud_dupes,
    )
    logger.info(
        "Cleaning: removed %s duplicate rows (%s of them fraud); %s rows remain",
        f"{summary.duplicates_removed:,}",
        summary.fraud_duplicates_removed,
        f"{summary.rows_after:,}",
    )
    return df, summary
