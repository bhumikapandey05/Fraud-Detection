"""Read-only access to the EXISTING evaluation reports of the served release.

Nothing is recomputed: the dashboard shows exactly the numbers recorded when the
release was evaluated. The snapshot is the one the API actually serves (with
``serving.frozen_snapshot: auto`` that follows promotions and rollbacks):

* the original Phase 2 release has validation AND held-out test results;
* a promoted retrained candidate has validation and recent-holdout results, but by
  design no test results - the test split is never used after Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config

FIGURES = (
    "pr_curves_validation.png",
    "pr_curve_test.png",
    "threshold_tradeoff.png",
    "confusion_matrices.png",
)


def snapshot_dir(cfg: Config, snapshot: str) -> Path:
    if not snapshot or snapshot == "auto" or "/" in snapshot or snapshot.startswith("."):
        raise ValueError(f"A concrete snapshot name is required, got {snapshot!r}")
    return cfg.paths.frozen_dir / snapshot


def _counts(m: dict) -> dict:
    return {k: int(m[k]) for k in ("tn", "fp", "fn", "tp")}


def load_performance(cfg: Config, snapshot: str) -> dict:
    root = snapshot_dir(cfg, snapshot)
    meta = json.loads((root / "model" / "model_metadata.json").read_text())
    test, val = meta.get("test_metrics"), meta["validation_metrics"]
    holdout = meta.get("holdout_metrics")
    th = meta["threshold"]
    cost_fp, cost_fn = th["cost_false_positive"], th["cost_false_negative"]

    def business(m: dict) -> dict:
        return {
            "missed_fraud": int(m["fn"]),
            "false_alarms": int(m["fp"]),
            "cost_missed_fraud": m["fn"] * cost_fn,
            "cost_false_alarms": m["fp"] * cost_fp,
            "total_cost": m["expected_cost"],
            "cost_without_model": m["cost_without_model"],
            "savings": m["cost_without_model"] - m["expected_cost"],
            "cost_reduction": m["cost_reduction"],
        }

    def summary(m: dict, ci: dict | None) -> dict:
        keys = ("pr_auc", "roc_auc", "precision", "recall", "f1", "alert_rate")
        return {
            "metrics": {k: m[k] for k in keys},
            "ci95": ci or {},
            "confusion_matrix": _counts(m),
            "n_transactions": int(m["n_samples"]),
            "n_fraud": int(m["n_positive"]),
            "business": business(m),
        }

    gap_path = root / "reports" / "validation_vs_test.json"
    gap = json.loads(gap_path.read_text()) if gap_path.is_file() else None
    comparison_path = root / "reports" / "model_comparison.csv"
    comparison = (
        pd.read_csv(comparison_path).to_dict(orient="records") if comparison_path.is_file() else []
    )
    return {
        "snapshot": snapshot,
        "kind": meta.get("kind", "release"),
        "model": meta["display_name"],
        "threshold": th,
        "evaluated_at": meta["created_at"],
        "test": summary(test, meta.get("test_ci95")) if test else None,
        "validation": summary(val, meta.get("validation_ci95")),
        "holdout": summary(holdout, None) if holdout else None,
        "model_comparison": comparison,
        "validation_vs_test": None
        if gap is None
        else {
            "difference": gap["difference_test_minus_validation"],
            "bootstrap_ci95_difference": gap["bootstrap_ci95_difference"],
            "fisher_exact_p_values": gap["fisher_exact_p_values"],
            "any_significant": gap["any_significant"],
        },
        "figures": [f for f in FIGURES if (root / "reports" / "figures" / f).is_file()],
    }


def figure_path(cfg: Config, snapshot: str, name: str) -> Path | None:
    if name not in FIGURES:  # whitelist: never build paths from user input
        return None
    path = snapshot_dir(cfg, snapshot) / "reports" / "figures" / name
    return path if path.is_file() else None
