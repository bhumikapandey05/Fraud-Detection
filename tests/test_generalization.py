import numpy as np
import pytest

from fraud_detection.analysis.generalization import (
    compare_validation_test,
    generalization_markdown,
    write_generalization_report,
)

COSTS = {"cost_false_positive": 5.0, "cost_false_negative": 120.0}


def _split(seed, n=4000, shift=0.4):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    scores = np.clip(rng.normal(0.3 + shift * y, 0.12), 0, 1)
    return y, scores


def test_identical_splits_show_no_difference():
    y, s = _split(0)
    r = compare_validation_test(y, s, y, s, 0.5, n_resamples=200, **COSTS)
    assert all(v == 0 for v in r["difference_test_minus_validation"].values())
    assert r["fisher_exact_p_values"]["recall"] == pytest.approx(1.0)
    assert not r["any_significant"]


def test_large_degradation_is_flagged_significant():
    y_val, s_val = _split(1, shift=0.5)
    y_test, s_test = _split(2, shift=0.05)  # test scores barely separate the classes
    r = compare_validation_test(y_val, s_val, y_test, s_test, 0.5, n_resamples=300, **COSTS)
    assert r["difference_test_minus_validation"]["recall"] < -0.3
    assert r["statistically_significant_at_5pct"]["pr_auc"]
    assert r["statistically_significant_at_5pct"]["recall"]
    assert r["bootstrap_ci95_difference"]["pr_auc"][1] < 0


def test_analysis_is_seeded_and_uses_given_threshold():
    y_val, s_val = _split(3)
    y_test, s_test = _split(4)
    a = compare_validation_test(y_val, s_val, y_test, s_test, 0.42, n_resamples=100, **COSTS)
    b = compare_validation_test(y_val, s_val, y_test, s_test, 0.42, n_resamples=100, **COSTS)
    assert a == b
    assert a["threshold"] == a["validation"]["threshold"] == a["test"]["threshold"] == 0.42


def test_report_files(tmp_path):
    y, s = _split(5)
    r = compare_validation_test(y, s, y, s, 0.5, n_resamples=50, **COSTS)
    paths = write_generalization_report(r, "Test Model", tmp_path)
    text = paths["markdown"].read_text()
    assert "Test Model" in text and "None of the differences" in text
    assert paths["json"].is_file()
    assert "statistically significant" in generalization_markdown(r, "x")
