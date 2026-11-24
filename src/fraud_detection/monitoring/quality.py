"""Data-quality checks on incoming transactions (pandas only, no Evidently needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data.schema import FEATURE_COLUMNS, TIME_COLUMN


def data_quality(
    features: pd.DataFrame,
    train_low: dict[str, float],
    train_high: dict[str, float],
    rejected: dict | None = None,
) -> dict:
    """Summarise quality of the scored rows (+ rows rejected before scoring, if any).

    ``rejected`` comes from batch ingestion: rows that could not be scored at all
    (missing / non-numeric / impossible values). API requests with such values are
    refused with HTTP 422 and never reach the store.
    """
    n = len(features)
    rejected = rejected or {}
    n_rejected = int(rejected.get("n_rejected", 0))
    n_total = n + n_rejected
    values = features[FEATURE_COLUMNS]

    checked = [c for c in FEATURE_COLUMNS if c != TIME_COLUMN]  # Time grows by design
    low = pd.Series({c: train_low[c] for c in checked})
    high = pd.Series({c: train_high[c] for c in checked})
    outside = (values[checked] < low) | (values[checked] > high)
    per_column = outside.sum()
    duplicates = int(values.duplicated().sum())

    return {
        "n_scored": n,
        "n_rejected": n_rejected,
        "rejected_share": n_rejected / n_total if n_total else 0.0,
        "rejection_reasons": rejected.get("reasons", {}),
        "missing_cells": int(values.isna().sum().sum()),
        "out_of_range_rows": int(outside.any(axis=1).sum()),
        "out_of_range_share": float(outside.any(axis=1).mean()) if n else 0.0,
        "out_of_range_by_column": {c: int(v) for c, v in per_column.items() if v > 0},
        "duplicate_rows": duplicates,
        "duplicate_share": duplicates / n if n else 0.0,
        "zero_amount_share": float((values["Amount"] == 0).mean()) if n else 0.0,
        "amount_median": float(np.median(values["Amount"])) if n else None,
    }
