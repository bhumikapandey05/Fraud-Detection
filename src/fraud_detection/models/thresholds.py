"""Choose the fraud decision threshold on validation data.

A model outputs a fraud *score* between 0 and 1. The threshold turns it into a
decision: flag the transaction if ``score >= threshold``. Lowering it catches
more fraud (fewer false negatives) but raises more false alarms (more false
positives). The default 0.5 is arbitrary - especially for class-weighted
models, whose scores are shifted upwards on purpose.

Two strategies are supported:

* ``cost``   - pick the threshold with the lowest total cost
  ``cost_false_negative * FN + cost_false_positive * FP``. This states the
  business trade-off explicitly ("a missed fraud costs ~24 false alarms").
* ``f_beta`` - pick the threshold with the highest F-beta score
  (``beta = 2`` weights recall twice as much as precision).

The threshold is tuned on the validation split only; the test split is never
used for any decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.config import ThresholdConfig


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    method: str
    objective: float  # total cost (lower is better) or F-beta (higher is better)


def threshold_curve(y_true, scores, beta: float = 2.0) -> pd.DataFrame:
    """Confusion counts and metrics at every distinct score, highest threshold first.

    Vectorised: sort once, then use cumulative sums, so it runs in O(n log n)
    instead of re-scoring the whole split for each candidate threshold.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        raise ValueError("Threshold tuning needs at least one positive example")

    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # Last index of each run of equal scores: predicting "score >= t" includes all ties.
    last = np.r_[np.flatnonzero(np.diff(s_sorted)), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = np.cumsum(1 - y_sorted)[last]
    positives, negatives = int(y.sum()), int(len(y) - y.sum())

    precision = tp / (tp + fp)
    recall = tp / positives
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        b2 = beta**2
        f_beta = np.where(
            precision + recall > 0,
            (1 + b2) * precision * recall / (b2 * precision + recall),
            0.0,
        )
    return pd.DataFrame(
        {
            "threshold": s_sorted[last],
            "tp": tp,
            "fp": fp,
            "fn": positives - tp,
            "tn": negatives - fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f_beta": f_beta,
            "alert_rate": (tp + fp) / len(y),
        }
    )


def add_costs(curve: pd.DataFrame, cfg: ThresholdConfig) -> pd.DataFrame:
    return curve.assign(
        expected_cost=cfg.cost_false_negative * curve["fn"] + cfg.cost_false_positive * curve["fp"]
    )


def tune_threshold(y_true, scores, cfg: ThresholdConfig) -> ThresholdResult:
    """Pick the best threshold on (validation) data according to ``cfg.method``.

    Ties are broken towards the *highest* threshold, i.e. fewer alerts.
    """
    curve = add_costs(threshold_curve(y_true, scores, beta=cfg.beta), cfg)
    if cfg.method == "cost":
        best = int(np.argmin(curve["expected_cost"].to_numpy()))
        objective = float(curve["expected_cost"].iloc[best])
    else:
        best = int(np.argmax(curve["f_beta"].to_numpy()))
        objective = float(curve["f_beta"].iloc[best])
    return ThresholdResult(
        threshold=float(curve["threshold"].iloc[best]), method=cfg.method, objective=objective
    )


def threshold_table(
    y_true, scores, cfg: ThresholdConfig, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9), extra=()
) -> pd.DataFrame:
    """Trade-off at a few readable thresholds (for the report)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    rows = []
    for t in sorted({*thresholds, *extra}):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(y.sum()) - tp
        rows.append(
            {
                "threshold": round(float(t), 4),
                "flagged": tp + fp,
                "fraud_caught": tp,
                "fraud_missed": fn,
                "false_alarms": fp,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / y.sum(), 4),
                "expected_cost": cfg.cost_false_negative * fn + cfg.cost_false_positive * fp,
            }
        )
    return pd.DataFrame(rows)
