"""Model performance on transactions whose true outcome is now known."""

from __future__ import annotations

import pandas as pd

from fraud_detection.models.metrics import classification_metrics


def labeled_performance(
    window: pd.DataFrame,
    threshold: float,
    cost_false_positive: float,
    cost_false_negative: float,
    min_fraud_labels: int,
) -> dict:
    """Metrics at the served threshold, using the scores that were actually served.

    With very few labelled frauds the numbers are noise, so below
    ``min_fraud_labels`` only the counts are reported.
    """
    labeled = window.dropna(subset=["label"])
    n_labeled, n_fraud = len(labeled), int(labeled["label"].sum()) if len(labeled) else 0
    base = {
        "n_labeled": n_labeled,
        "n_fraud_labels": n_fraud,
        "label_coverage": n_labeled / len(window) if len(window) else 0.0,
    }
    if n_fraud < min_fraud_labels or n_fraud == n_labeled:
        return {
            **base,
            "status": "insufficient_labels",
            "message": f"{n_fraud} labelled frauds (need {min_fraud_labels}) - not evaluated",
        }
    metrics = classification_metrics(
        labeled["label"].astype(int),
        labeled["fraud_score"],
        threshold,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
    )
    return {**base, "status": "evaluated", **metrics}
