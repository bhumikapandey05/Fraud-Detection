"""Leakage-safe train / validation / test splitting.

Default strategy: **time-based**. Rows are ordered by ``Time`` and cut into
three consecutive blocks (oldest -> train, middle -> validation, newest ->
test). This mirrors production, where a model trained on past transactions
scores future ones, and prevents information from the future leaking into
training.

A ``"stratified"`` random split is also available as a comparison baseline; it
keeps the fraud rate identical across splits but ignores time order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import SplitConfig
from fraud_detection.data.schema import TARGET_COLUMN, TIME_COLUMN
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def summary(
        self, target: str = TARGET_COLUMN, time_column: str = TIME_COLUMN
    ) -> dict[str, dict]:
        total = sum(len(df) for df in self.as_dict().values())
        out = {}
        for name, df in self.as_dict().items():
            positives = int(df[target].sum())
            out[name] = {
                "rows": len(df),
                "share_of_rows": round(len(df) / total, 4) if total else 0.0,
                "fraud_cases": positives,
                "fraud_rate": round(positives / len(df), 6) if len(df) else 0.0,
                "time_start": float(df[time_column].min()) if len(df) else None,
                "time_end": float(df[time_column].max()) if len(df) else None,
            }
        return out


def time_based_split(
    df: pd.DataFrame,
    train_size: float,
    validation_size: float,
    time_column: str = TIME_COLUMN,
) -> DataSplits:
    """Split chronologically using time *cut-off points*, not row positions.

    Cutting on timestamps (``train: Time < t1``, ``validation: t1 <= Time < t2``,
    ``test: Time >= t2``) guarantees that transactions sharing a timestamp never
    straddle two splits, so every split is strictly later than the previous one.
    """
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create three splits")

    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    times = ordered[time_column]
    n = len(ordered)
    t1 = times.iloc[int(n * train_size)]
    t2 = times.iloc[int(n * (train_size + validation_size))]

    train = ordered[times < t1]
    validation = ordered[(times >= t1) & (times < t2)]
    test = ordered[times >= t2]
    logger.info("Time split cut-offs: t1=%.0fs, t2=%.0fs", t1, t2)
    return DataSplits(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def stratified_split(
    df: pd.DataFrame,
    train_size: float,
    validation_size: float,
    random_state: int,
    target: str = TARGET_COLUMN,
) -> DataSplits:
    """Random split that preserves the fraud rate in every split."""
    train, holdout = train_test_split(
        df, train_size=train_size, stratify=df[target], random_state=random_state
    )
    val_share = validation_size / (1 - train_size)
    validation, test = train_test_split(
        holdout, train_size=val_share, stratify=holdout[target], random_state=random_state
    )
    return DataSplits(
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split_data(
    df: pd.DataFrame,
    cfg: SplitConfig,
    target: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> DataSplits:
    """Split ``df`` according to the configured strategy."""
    if cfg.strategy == "time":
        splits = time_based_split(df, cfg.train_size, cfg.validation_size, time_column)
    else:
        splits = stratified_split(df, cfg.train_size, cfg.validation_size, cfg.random_state, target)
    for name, stats in splits.summary(target, time_column).items():
        logger.info(
            "%-10s rows=%8s  fraud=%4d  fraud_rate=%.4f%%",
            name,
            f"{stats['rows']:,}",
            stats["fraud_cases"],
            100 * stats["fraud_rate"],
        )
    return splits
