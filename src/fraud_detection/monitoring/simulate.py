"""Synthetic "production" batches for demonstrating and testing monitoring.

All batches are generated from the TRAINING split only (never validation or test),
by resampling transactions, adding small noise, moving them to later days, and
applying one scenario transform. Labels come from the source rows.

Because the rows come from data the model was trained on, absolute performance
numbers on these batches are optimistic. The batches exist to show that monitoring
reacts correctly to known changes - not to estimate real-world accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.data.schema import AMOUNT_COLUMN, FEATURE_COLUMNS, PCA_FEATURES, TIME_COLUMN

SECONDS_PER_DAY = 24 * 3600

SCENARIOS = {
    "stable": "Same population as training (resampled + small noise). Expect no alerts.",
    "amount_shift": "All amounts x2.5 (e.g. price changes / new merchant mix). Expect drift "
    "in Amount.",
    "feature_shift": "Eight anonymized components shifted by one standard deviation. Expect "
    "a data-drift alert.",
    "quality_issues": "2% rows with a missing value, 2% far outside the training range, 3% "
    "duplicates. Expect data-quality alerts.",
    "concept_drift": "Fraudsters change behaviour: fraud transactions look much more like "
    "legitimate ones (3x the usual fraud rate). Expect recall to drop.",
}
SHIFTED_COMPONENTS = ["V1", "V2", "V5", "V6", "V8", "V13", "V15", "V19"]


def generate_batch(
    train: pd.DataFrame,
    scenario: str,
    n_rows: int,
    seed: int,
    day: int = 1,
    batch_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transactions with ``transaction_id``, labels with ``transaction_id, label``)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    batch_id = batch_id or f"{scenario}-day{day}"

    fraud_rate = float(train["Class"].mean()) * (3 if scenario == "concept_drift" else 1)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    frauds = train[train["Class"] == 1]
    legit = train[train["Class"] == 0]
    rows = (
        pd.concat(
            [
                frauds.sample(n_fraud, replace=True, random_state=seed),
                legit.sample(n_rows - n_fraud, replace=True, random_state=seed + 1),
            ]
        )
        .sample(frac=1, random_state=seed + 2)
        .reset_index(drop=True)
    )
    labels = rows["Class"].astype(int).to_numpy()
    x = rows[FEATURE_COLUMNS].copy()

    # Small noise so rows are not exact copies of training rows.
    stds = train[PCA_FEATURES].std()
    x[PCA_FEATURES] += rng.normal(0, 0.05, (len(x), len(PCA_FEATURES))) * stds.to_numpy()
    x[AMOUNT_COLUMN] = (x[AMOUNT_COLUMN] * rng.lognormal(0, 0.05, len(x))).round(2)
    # The noise must not create out-of-range values by itself: only scenarios change data.
    cols = [*PCA_FEATURES, AMOUNT_COLUMN]
    x[cols] = x[cols].clip(train[cols].min(), train[cols].max(), axis=1)
    # Later days, same hour of day as the source transaction.
    x[TIME_COLUMN] = (x[TIME_COLUMN] % SECONDS_PER_DAY) + SECONDS_PER_DAY * (day + 2)

    if scenario == "amount_shift":
        x[AMOUNT_COLUMN] = (x[AMOUNT_COLUMN] * 2.5).round(2)
    elif scenario == "feature_shift":
        x[SHIFTED_COMPONENTS] += stds[SHIFTED_COMPONENTS].to_numpy()
    elif scenario == "quality_issues":
        n = len(x)
        bad = rng.choice(n, int(0.02 * n), replace=False)
        x.loc[bad, "V3"] = np.nan
        odd = rng.choice(np.setdiff1d(np.arange(n), bad), int(0.02 * n), replace=False)
        x.loc[odd, "V7"] = 300.0
        dup = rng.choice(n, int(0.03 * n), replace=False)
        x = pd.concat([x, x.iloc[dup]], ignore_index=True)
        labels = np.concatenate([labels, labels[dup]])
    elif scenario == "concept_drift":
        legit_mean = legit[PCA_FEATURES].mean().to_numpy()
        is_fraud = labels == 1
        x.loc[is_fraud, PCA_FEATURES] = legit_mean + 0.35 * (
            x.loc[is_fraud, PCA_FEATURES].to_numpy() - legit_mean
        )

    ids = [f"{batch_id}-{i:06d}" for i in range(len(x))]
    x.insert(0, "transaction_id", ids)
    return x.reset_index(drop=True), pd.DataFrame({"transaction_id": ids, "label": labels})
