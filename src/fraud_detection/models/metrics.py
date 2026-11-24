"""Evaluation metrics suited to rare-event (fraud) classification.

Accuracy is deliberately not reported as a headline number: predicting
"never fraud" is 99.8% accurate and useless. Instead we report:

* **PR-AUC** (average precision): ranking quality focused on the fraud class.
  A random model scores roughly the fraud rate (~0.0013 here).
* **ROC-AUC**: overall ranking quality (can look optimistic under imbalance).
* **Precision / recall / F1** at the chosen decision threshold.
* **Confusion matrix** counts and an **expected cost** using the configured
  business costs of false alarms vs missed fraud.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tn": int((~y_true & ~y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tp": int((y_true & y_pred).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(
    y_true,
    scores,
    threshold: float,
    cost_false_positive: float | None = None,
    cost_false_negative: float | None = None,
) -> dict[str, float | int]:
    """All headline metrics for one set of scores at one decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    counts = confusion_counts(y_true, scores >= threshold)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        **counts,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
        "alert_rate": _safe_div(tp + fp, len(y_true)),
    }
    if cost_false_positive is not None and cost_false_negative is not None:
        cost = cost_false_positive * fp + cost_false_negative * fn
        no_model_cost = cost_false_negative * int(y_true.sum())  # miss every fraud
        metrics["expected_cost"] = float(cost)
        metrics["cost_without_model"] = float(no_model_cost)
        metrics["cost_reduction"] = _safe_div(no_model_cost - cost, no_model_cost)
    return metrics


def bootstrap_confidence_intervals(
    y_true,
    scores,
    threshold: float,
    n_resamples: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Percentile bootstrap CIs for PR-AUC, ROC-AUC, precision and recall.

    With only ~50 fraud cases in the validation/test splits, a single number
    can be misleading; the interval shows how much it could move by chance.
    """
    if n_resamples <= 0:
        return {}
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "precision": [], "recall": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y, s = y_true[idx], scores[idx]
        if y.min() == y.max():  # a resample with a single class has no defined AUC
            continue
        pred = s >= threshold
        tp = int((pred & (y == 1)).sum())
        samples["pr_auc"].append(average_precision_score(y, s))
        samples["roc_auc"].append(roc_auc_score(y, s))
        samples["precision"].append(_safe_div(tp, int(pred.sum())))
        samples["recall"].append(_safe_div(tp, int(y.sum())))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))]
        for k, v in samples.items()
        if v
    }
