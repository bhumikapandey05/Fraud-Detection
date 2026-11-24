"""How much does performance change from validation to test? (read-only)

The model and threshold were chosen on the validation split, so the test split
is the unbiased estimate. This module quantifies the gap between the two using
the FROZEN detector and threshold - nothing is refitted or re-tuned:

* metrics on both splits at the same threshold,
* a bootstrap 95% interval for the test-minus-validation difference in
  PR-AUC and recall (each split resampled independently),
* Fisher exact tests for recall, precision and false-alarm rate
  (are the 2x2 count tables plausibly from the same underlying rate?).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import average_precision_score

from fraud_detection.models.metrics import classification_metrics

SIGNIFICANCE_LEVEL = 0.05
COMPARED = ("pr_auc", "roc_auc", "precision", "recall", "f1", "alert_rate", "cost_reduction")


def _fisher_p(a_hits: int, a_total: int, b_hits: int, b_total: int) -> float:
    table = [[a_hits, a_total - a_hits], [b_hits, b_total - b_hits]]
    return float(fisher_exact(table)[1])


def compare_validation_test(
    y_val,
    val_scores,
    y_test,
    test_scores,
    threshold: float,
    cost_false_positive: float,
    cost_false_negative: float,
    n_resamples: int = 2000,
    random_state: int = 42,
) -> dict:
    """Quantify the validation -> test change for one frozen model + threshold."""
    costs = {"cost_false_positive": cost_false_positive, "cost_false_negative": cost_false_negative}
    y_val, y_test = np.asarray(y_val).astype(int), np.asarray(y_test).astype(int)
    val_scores, test_scores = np.asarray(val_scores), np.asarray(test_scores)
    val = classification_metrics(y_val, val_scores, threshold, **costs)
    test = classification_metrics(y_test, test_scores, threshold, **costs)

    rng = np.random.default_rng(random_state)
    pr_diffs, recall_diffs = [], []
    for _ in range(n_resamples):
        vi = rng.integers(0, len(y_val), len(y_val))
        ti = rng.integers(0, len(y_test), len(y_test))
        yv, yt = y_val[vi], y_test[ti]
        if yv.min() == yv.max() or yt.min() == yt.max():
            continue
        pr_diffs.append(
            average_precision_score(yt, test_scores[ti])
            - average_precision_score(yv, val_scores[vi])
        )
        recall_diffs.append(
            (test_scores[ti][yt == 1] >= threshold).mean()
            - (val_scores[vi][yv == 1] >= threshold).mean()
        )

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    p_values = {
        "recall": _fisher_p(val["tp"], val["n_positive"], test["tp"], test["n_positive"]),
        "precision": _fisher_p(
            val["tp"], val["tp"] + val["fp"], test["tp"], test["tp"] + test["fp"]
        ),
        "false_alarm_rate": _fisher_p(
            val["fp"], val["fp"] + val["tn"], test["fp"], test["fp"] + test["tn"]
        ),
    }
    pr_ci, recall_ci = interval(pr_diffs), interval(recall_diffs)
    significant = {
        "pr_auc": not (pr_ci[0] <= 0 <= pr_ci[1]),
        "recall": p_values["recall"] < SIGNIFICANCE_LEVEL,
        "precision": p_values["precision"] < SIGNIFICANCE_LEVEL,
        "false_alarm_rate": p_values["false_alarm_rate"] < SIGNIFICANCE_LEVEL,
    }
    return {
        "threshold": float(threshold),
        "validation": val,
        "test": test,
        "difference_test_minus_validation": {k: test[k] - val[k] for k in COMPARED},
        "bootstrap_ci95_difference": {"pr_auc": pr_ci, "recall": recall_ci},
        "share_of_resamples_test_below_validation": float(np.mean(np.array(pr_diffs) < 0)),
        "fisher_exact_p_values": p_values,
        "statistically_significant_at_5pct": significant,
        "any_significant": any(significant.values()),
        "n_resamples": len(pr_diffs),
    }


def generalization_markdown(result: dict, model_name: str) -> str:
    v, t = result["validation"], result["test"]
    d = result["difference_test_minus_validation"]
    ci = result["bootstrap_ci95_difference"]
    p = result["fisher_exact_p_values"]
    rows = [
        ("PR-AUC", "pr_auc", "{:.3f}"),
        ("ROC-AUC", "roc_auc", "{:.3f}"),
        ("Precision", "precision", "{:.3f}"),
        ("Recall", "recall", "{:.3f}"),
        ("F1", "f1", "{:.3f}"),
        ("Cost reduction vs no model", "cost_reduction", "{:.1%}"),
    ]
    table = "\n".join(
        f"| {label} | {fmt.format(v[k])} | {fmt.format(t[k])} | {d[k]:+.3f} |"
        for label, k, fmt in rows
    )
    counts = (
        f"| Fraud caught | {v['tp']} of {v['n_positive']} | {t['tp']} of {t['n_positive']} | |\n"
        f"| Fraud missed | {v['fn']} | {t['fn']} | |\n"
        f"| False alarms | {v['fp']} of {v['fp'] + v['tn']:,} "
        f"| {t['fp']} of {t['fp'] + t['tn']:,} | |\n"
        f"| Expected cost | ${v['expected_cost']:,.0f} | ${t['expected_cost']:,.0f} | |"
    )
    share_below = result["share_of_resamples_test_below_validation"]
    verdict = (
        "At least one difference is statistically significant at the 5% level."
        if result["any_significant"]
        else "**None of the differences is statistically significant at the 5% level.** "
        "The direction is consistent (test is lower), but with ~50 fraud cases per split "
        "the gap cannot be separated from chance."
    )
    return f"""# Validation → test performance: {model_name}

Frozen model and threshold ({result["threshold"]:.4f}, tuned on validation). Nothing
was refitted or re-tuned for this analysis.

| Metric | Validation | Test | Test − validation |
|---|---:|---:|---:|
{table}
{counts}

## Is the drop real or chance?

| Check | Result |
|---|---|
| PR-AUC difference, bootstrap 95% CI | {ci["pr_auc"][0]:+.3f} to {ci["pr_auc"][1]:+.3f} |
| Recall difference, bootstrap 95% CI | {ci["recall"][0]:+.3f} to {ci["recall"][1]:+.3f} |
| Resamples where test PR-AUC < validation | {share_below:.0%} |
| Recall: Fisher exact p-value | {p["recall"]:.3f} |
| Precision: Fisher exact p-value | {p["precision"]:.3f} |
| False-alarm rate: Fisher exact p-value | {p["false_alarm_rate"]:.3f} |

{verdict}

## Likely reasons for the drop

1. **Selection optimism ("winner's curse").** The model *and* its threshold were chosen
   because they scored best on validation, so the validation numbers are slightly
   flattering. Test was never used for any decision, so it is the honest estimate.
2. **Time drift.** The test split is the latest period; fraud patterns change over time.
3. **Small samples.** With ~50 frauds per split, each fraud case is worth ~2 points of
   recall, so a handful of hard cases moves the numbers a lot.

## What we report

The **test** numbers are the performance estimate for this model. The validation numbers
are kept only to document how the model and threshold were selected. This gap should be
monitored after deployment (Phase 4), especially recall and alert volume.
"""


def write_generalization_report(result: dict, model_name: str, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out_dir / "validation_vs_test.md",
        "json": out_dir / "validation_vs_test.json",
    }
    paths["markdown"].write_text(generalization_markdown(result, model_name))
    paths["json"].write_text(json.dumps(result, indent=2))
    return paths
