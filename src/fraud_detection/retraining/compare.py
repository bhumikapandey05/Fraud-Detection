"""Candidate-vs-champion promotion criteria (all must pass for eligibility)."""

from __future__ import annotations

from fraud_detection.config import PromotionCriteria


def evaluate_criteria(
    candidate_val: dict,
    champion_val: dict,
    candidate_holdout: dict | None,
    champion_holdout: dict | None,
    c: PromotionCriteria,
) -> dict:
    checks = [
        {
            "name": "validation_pr_auc",
            "description": f"Validation PR-AUC no more than {c.max_validation_pr_auc_drop} below "
            "the champion",
            "candidate": candidate_val["pr_auc"],
            "champion": champion_val["pr_auc"],
            "passed": candidate_val["pr_auc"]
            >= champion_val["pr_auc"] - c.max_validation_pr_auc_drop,
        },
        {
            "name": "validation_recall",
            "description": f"Validation recall (own tuned threshold) no more than "
            f"{c.max_validation_recall_drop} below the champion",
            "candidate": candidate_val["recall"],
            "champion": champion_val["recall"],
            "passed": candidate_val["recall"]
            >= champion_val["recall"] - c.max_validation_recall_drop,
        },
        {
            "name": "validation_cost",
            "description": f"Validation expected cost at most {c.max_validation_cost_increase:.0%} "
            "above the champion",
            "candidate": candidate_val["expected_cost"],
            "champion": champion_val["expected_cost"],
            "passed": candidate_val["expected_cost"]
            <= champion_val["expected_cost"] * (1 + c.max_validation_cost_increase),
        },
    ]
    if candidate_holdout is not None and champion_holdout is not None:
        checks.append(
            {
                "name": "holdout_pr_auc",
                "description": f"Recent-holdout PR-AUC at least {c.min_holdout_pr_auc_gain:+} vs "
                "the champion",
                "candidate": candidate_holdout["pr_auc"],
                "champion": champion_holdout["pr_auc"],
                "passed": candidate_holdout["pr_auc"]
                >= champion_holdout["pr_auc"] + c.min_holdout_pr_auc_gain,
            }
        )
    elif c.require_holdout:
        checks.append(
            {
                "name": "holdout_pr_auc",
                "description": "A recent labelled holdout with enough frauds is required",
                "candidate": None,
                "champion": None,
                "passed": False,
            }
        )
    return {"eligible": all(ch["passed"] for ch in checks), "checks": checks}
