import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import ThresholdConfig
from fraud_detection.models.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    confusion_counts,
)
from fraud_detection.models.thresholds import (
    add_costs,
    threshold_curve,
    threshold_table,
    tune_threshold,
)

Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
S = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.65, 0.8, 0.9, 0.95])


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0, 1).round(3)  # rounding creates ties
    return y, scores


# --- metrics ---------------------------------------------------------------------
def test_confusion_counts():
    assert confusion_counts([0, 1, 1, 0], [0, 1, 0, 1]) == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_classification_metrics_hand_computed():
    m = classification_metrics(Y, S, 0.6, cost_false_positive=1, cost_false_negative=10)
    # flagged: 0.6(neg), 0.65, 0.8, 0.9 (pos), 0.95(neg) -> tp=3 fp=2 fn=0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 2, 0, 5)
    assert m["precision"] == pytest.approx(0.6)
    assert m["recall"] == 1.0
    assert m["f1"] == pytest.approx(0.75)
    assert m["expected_cost"] == 2.0
    assert m["cost_without_model"] == 30.0
    assert m["alert_rate"] == 0.5


def test_perfect_ranking_scores_one():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0 and m["f1"] == 1.0


def test_no_alerts_gives_zero_precision_not_error():
    m = classification_metrics([0, 1], [0.1, 0.2], 0.99)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_bootstrap_is_seeded_and_brackets_estimate(noisy):
    y, s = noisy
    a = bootstrap_confidence_intervals(y, s, 0.5, n_resamples=200, random_state=3)
    b = bootstrap_confidence_intervals(y, s, 0.5, n_resamples=200, random_state=3)
    assert a == b
    point = classification_metrics(y, s, 0.5)
    for key in ("pr_auc", "roc_auc", "precision", "recall"):
        low, high = a[key]
        assert low <= point[key] <= high
    assert bootstrap_confidence_intervals(y, s, 0.5, n_resamples=0) == {}


# --- thresholds ------------------------------------------------------------------
def test_threshold_curve_matches_brute_force(noisy):
    y, s = noisy
    curve = threshold_curve(y, s)
    assert curve["threshold"].is_monotonic_decreasing
    assert curve["threshold"].is_unique
    for row in curve.sample(40, random_state=0).itertuples():
        expected = confusion_counts(y, s >= row.threshold)
        assert (row.tp, row.fp, row.fn, row.tn) == (
            expected["tp"],
            expected["fp"],
            expected["fn"],
            expected["tn"],
        )


def test_cost_tuning_hand_example():
    # FN is 10x worse than FP: flagging 0.6..0.95 (tp=3, fp=2, cost 2) beats
    # flagging only 0.65..0.95 (tp=3, fp=1, cost 1)? -> the latter is cheaper.
    result = tune_threshold(Y, S, ThresholdConfig(cost_false_positive=1, cost_false_negative=10))
    assert result.threshold == 0.65
    assert result.objective == 1.0


def test_cost_tuning_matches_brute_force(noisy):
    y, s = noisy
    cfg = ThresholdConfig(cost_false_positive=5, cost_false_negative=120)
    result = tune_threshold(y, s, cfg)
    costs = {
        t: 120 * ((s < t) & (y == 1)).sum() + 5 * ((s >= t) & (y == 0)).sum() for t in np.unique(s)
    }
    assert result.objective == min(costs.values())
    assert result.threshold == max(t for t, c in costs.items() if c == result.objective)


def test_higher_miss_cost_never_raises_threshold(noisy):
    y, s = noisy
    thresholds = [
        tune_threshold(
            y, s, ThresholdConfig(cost_false_positive=1, cost_false_negative=fn)
        ).threshold
        for fn in (1, 5, 20, 100, 1000)
    ]
    assert thresholds == sorted(thresholds, reverse=True)


def test_f_beta_tuning(noisy):
    y, s = noisy
    result = tune_threshold(y, s, ThresholdConfig(method="f_beta", beta=2.0))
    curve = threshold_curve(y, s, beta=2.0)
    assert result.objective == pytest.approx(curve["f_beta"].max())
    assert result.method == "f_beta"


def test_tie_break_prefers_higher_threshold():
    # Both thresholds give cost 1: choose the one that raises fewer alerts.
    y = np.array([1, 0, 0])
    s = np.array([0.9, 0.5, 0.1])
    cfg = ThresholdConfig(cost_false_positive=1, cost_false_negative=1)
    assert tune_threshold(y, s, cfg).threshold == 0.9


def test_threshold_needs_positives():
    with pytest.raises(ValueError):
        threshold_curve([0, 0], [0.1, 0.2])


def test_threshold_table_and_costs():
    cfg = ThresholdConfig(cost_false_positive=1, cost_false_negative=10)
    table = threshold_table(Y, S, cfg, thresholds=(0.5,), extra=(0.65,))
    assert list(table["threshold"]) == [0.5, 0.65]
    row = table.iloc[1]
    assert (row.fraud_caught, row.false_alarms, row.fraud_missed) == (3, 1, 0)
    assert row.expected_cost == 1
    curve = add_costs(threshold_curve(Y, S), cfg)
    assert isinstance(curve, pd.DataFrame) and "expected_cost" in curve
