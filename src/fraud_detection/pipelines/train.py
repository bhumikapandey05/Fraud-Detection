"""Training pipeline (Phase 2).

Steps:

1. **Verify** the prepared splits: schema, counts vs Phase 1, no leakage.
2. **Train** every configured model on the training split only. Each model is a
   full pipeline (Phase 1 preprocessing + classifier) with training-only class
   weights.
3. **Score** the validation split and **tune** each model's decision threshold
   on it (cost-based by default).
4. **Select** the best model by validation PR-AUC.
5. **Test once**: only the selected model, with its validation-tuned threshold,
   is scored on the held-out test split. No decision is made after this.
6. **Save** the inference pipeline + threshold + metadata.
7. **Report**: comparison table, PR curves, threshold trade-off, confusion
   matrices and a Markdown evaluation report.
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from fraud_detection.analysis import model_report as rpt
from fraud_detection.config import Config
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.data.verification import (
    SPLIT_NAMES,
    check_preprocessor_consistency,
    verify_prepared_data,
)
from fraud_detection.features.preprocessing import split_features_target
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import FraudDetector, save_detector
from fraud_detection.models.factory import MODEL_DISPLAY_NAMES, build_training_pipeline
from fraud_detection.models.metrics import bootstrap_confidence_intervals, classification_metrics
from fraud_detection.models.thresholds import (
    add_costs,
    threshold_curve,
    threshold_table,
    tune_threshold,
)

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.5


@dataclass
class CandidateResult:
    """Everything learned about one model on train + validation data."""

    name: str
    pipeline: Pipeline
    train_seconds: float
    val_scores: np.ndarray
    threshold: float
    val_metrics: dict
    val_metrics_default: dict
    val_ci: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY_NAMES[self.name]


def _cost_kwargs(cfg: Config) -> dict:
    return {
        "cost_false_positive": cfg.threshold.cost_false_positive,
        "cost_false_negative": cfg.threshold.cost_false_negative,
    }


def train_candidate(name: str, cfg: Config, X_train, y_train, X_val, y_val) -> CandidateResult:
    """Fit one model on train, tune its threshold on validation, and score it."""
    pipeline = build_training_pipeline(
        name,
        cfg.training.params.get(name),
        cfg.training.random_state,
        cfg.preprocessing,
        y_train=y_train,
    )
    logger.info("Training %s ...", MODEL_DISPLAY_NAMES[name])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    val_scores = pipeline.predict_proba(X_val)[:, 1]
    tuned = tune_threshold(y_val, val_scores, cfg.threshold)
    val_metrics = classification_metrics(y_val, val_scores, tuned.threshold, **_cost_kwargs(cfg))
    val_default = classification_metrics(y_val, val_scores, DEFAULT_THRESHOLD, **_cost_kwargs(cfg))
    val_ci = bootstrap_confidence_intervals(
        y_val,
        val_scores,
        tuned.threshold,
        n_resamples=cfg.training.bootstrap_samples,
        random_state=cfg.training.random_state,
    )
    logger.info(
        "%-20s val PR-AUC=%.4f ROC-AUC=%.4f | threshold=%.4f precision=%.3f recall=%.3f (%.1fs)",
        MODEL_DISPLAY_NAMES[name],
        val_metrics["pr_auc"],
        val_metrics["roc_auc"],
        tuned.threshold,
        val_metrics["precision"],
        val_metrics["recall"],
        train_seconds,
    )
    return CandidateResult(
        name=name,
        pipeline=pipeline,
        train_seconds=train_seconds,
        val_scores=val_scores,
        threshold=tuned.threshold,
        val_metrics=val_metrics,
        val_metrics_default=val_default,
        val_ci=val_ci,
    )


def select_best(candidates: list[CandidateResult], metric: str) -> CandidateResult:
    """Highest validation ``metric`` wins; ties keep the configured model order."""
    return max(candidates, key=lambda c: c.val_metrics[metric])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict[str, str]:
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(),
    }


def _model_params(pipeline: Pipeline) -> dict:
    params = pipeline.named_steps["model"].get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}


def run_train(cfg: Config) -> dict:
    """Execute the full Phase 2 training and evaluation pipeline."""
    target = cfg.dataset.target
    report_dir = cfg.paths.training_report_dir
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify data -------------------------------------------------------------
    verified = verify_prepared_data(cfg)
    splits = verified.splits
    X_train, y_train = split_features_target(splits.train, target)
    X_val, y_val = split_features_target(splits.validation, target)

    # 2-3. Train candidates and tune thresholds on validation ---------------------
    candidates = [
        train_candidate(name, cfg, X_train, y_train, X_val, y_val) for name in cfg.training.models
    ]
    preprocessor_consistent = check_preprocessor_consistency(
        candidates[0].pipeline.named_steps["preprocessor"],
        cfg.paths.artifacts_dir,
        X_val.head(2000),
    )
    if preprocessor_consistent is False:
        raise RuntimeError(
            "Pipeline preprocessing differs from the Phase 1 artifact - re-run "
            "`fraud-detection prepare`"
        )

    # 4. Select on validation ------------------------------------------------------
    best = select_best(candidates, cfg.training.selection_metric)
    logger.info(
        "Selected %s (validation %s = %.4f)",
        best.display_name,
        cfg.training.selection_metric,
        best.val_metrics[cfg.training.selection_metric],
    )

    # 5. Evaluate ONCE on the held-out test split ----------------------------------
    X_test, y_test = split_features_target(splits.test, target)
    test_scores = best.pipeline.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_scores, best.threshold, **_cost_kwargs(cfg))
    test_ci = bootstrap_confidence_intervals(
        y_test,
        test_scores,
        best.threshold,
        n_resamples=cfg.training.bootstrap_samples,
        random_state=cfg.training.random_state,
    )
    logger.info(
        "TEST  PR-AUC=%.4f ROC-AUC=%.4f precision=%.3f recall=%.3f (TP=%d FP=%d FN=%d)",
        test_metrics["pr_auc"],
        test_metrics["roc_auc"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["tp"],
        test_metrics["fp"],
        test_metrics["fn"],
    )

    # 6. Save the inference pipeline + threshold + metadata ------------------------
    detector = FraudDetector(best.pipeline, best.threshold, best.name, FEATURE_COLUMNS)
    metadata = {
        "model_name": best.name,
        "display_name": best.display_name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "random_state": cfg.training.random_state,
        "threshold": {
            "value": best.threshold,
            "tuned_on": "validation",
            "method": cfg.threshold.method,
            "cost_false_positive": cfg.threshold.cost_false_positive,
            "cost_false_negative": cfg.threshold.cost_false_negative,
            "beta": cfg.threshold.beta,
        },
        "selection": {
            "metric": cfg.training.selection_metric,
            "split": "validation",
            "candidates": {
                c.name: {
                    "pr_auc": c.val_metrics["pr_auc"],
                    "roc_auc": c.val_metrics["roc_auc"],
                    "threshold": c.threshold,
                    "precision": c.val_metrics["precision"],
                    "recall": c.val_metrics["recall"],
                    "f1": c.val_metrics["f1"],
                    "expected_cost": c.val_metrics["expected_cost"],
                }
                for c in candidates
            },
        },
        "hyperparameters": _model_params(best.pipeline),
        "class_imbalance": {
            "strategy": "class weights from training labels only; no resampling",
            "train_negative_to_positive_ratio": round(
                float((len(y_train) - y_train.sum()) / y_train.sum()), 2
            ),
        },
        "input_features": FEATURE_COLUMNS,
        "model_features": best.pipeline.named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist(),
        "validation_metrics": best.val_metrics,
        "validation_ci95": best.val_ci,
        "test_metrics": test_metrics,
        "test_ci95": test_ci,
        "data": {
            "splits": verified.summary,
            "files_sha256": {
                name: _file_sha256(cfg.paths.processed_dir / f"{name}.parquet")
                for name in SPLIT_NAMES
            },
            "verification": verified.report.to_dict(),
            "preprocessor_matches_phase1_artifact": preprocessor_consistent,
        },
        "environment": _library_versions(),
    }
    saved = save_detector(detector, metadata, cfg.paths.model_dir)
    logger.info("Saved detector to %s", saved["detector"])

    # 7. Reports -----------------------------------------------------------------------
    table = rpt.comparison_table(
        [
            {
                "model": c.display_name,
                "selected": c is best,
                "val": c.val_metrics,
                "val_default": c.val_metrics_default,
                "val_ci": c.val_ci,
                "train_seconds": c.train_seconds,
            }
            for c in candidates
        ]
    )
    table.to_csv(report_dir / "model_comparison.csv", index=False)

    val_curve = add_costs(
        threshold_curve(y_val, best.val_scores, beta=cfg.threshold.beta), cfg.threshold
    )
    tradeoff = threshold_table(y_val, best.val_scores, cfg.threshold, extra=(best.threshold,))
    tradeoff.to_csv(report_dir / "threshold_tradeoff_validation.csv", index=False)

    figures = [
        rpt.plot_pr_curves(
            {
                c.display_name: {"y": y_val, "scores": c.val_scores, "threshold": c.threshold}
                for c in candidates
            },
            fig_dir / "pr_curves_validation.png",
            "Precision-recall curves (validation)",
        ),
        rpt.plot_pr_curves(
            {
                f"{best.display_name} - validation": {
                    "y": y_val,
                    "scores": best.val_scores,
                    "threshold": best.threshold,
                },
                f"{best.display_name} - test": {
                    "y": y_test,
                    "scores": test_scores,
                    "threshold": best.threshold,
                },
            },
            fig_dir / "pr_curve_test.png",
            f"Selected model: validation vs held-out test ({best.display_name})",
        ),
        rpt.plot_threshold_tradeoff(
            val_curve,
            best.threshold,
            fig_dir / "threshold_tradeoff.png",
            f"Threshold trade-off on validation ({best.display_name})",
        ),
        rpt.plot_confusion_matrices(
            {
                "Validation": best.val_metrics,
                "Test (held out)": test_metrics,
            },
            fig_dir / "confusion_matrices.png",
        ),
    ]

    report_path = _write_evaluation_report(
        cfg,
        verified,
        candidates,
        best,
        table,
        tradeoff,
        test_metrics,
        test_ci,
        preprocessor_consistent,
        figures,
        report_dir,
    )
    logger.info("Evaluation report written to %s", report_path)

    return {
        "selected_model": best.name,
        "threshold": best.threshold,
        "validation": {c.name: c.val_metrics for c in candidates},
        "test": test_metrics,
        "test_ci95": test_ci,
        "artifacts": {k: str(v) for k, v in saved.items()},
        "report": str(report_path),
    }


# ------------------------------------------------------------------------------------ #
# Report text
# ------------------------------------------------------------------------------------ #
def _ci(ci: dict, key: str) -> str:
    return f" (95% CI {ci[key][0]:.3f}–{ci[key][1]:.3f})" if key in ci else ""


def _write_evaluation_report(
    cfg, verified, candidates, best, table, tradeoff, test, test_ci, consistent, figures, out_dir
) -> Path:
    th = cfg.threshold
    summary = verified.summary
    split_rows = "\n".join(
        f"| {name} | {s['rows']:,} | {s['fraud_cases']} | {100 * s['fraud_rate']:.3f}% |"
        for name, s in summary.items()
    )
    ranked = sorted(candidates, key=lambda c: c.val_metrics["pr_auc"], reverse=True)
    gap_sentence = (
        f" The validation PR-AUC gap between {ranked[0].display_name} and "
        f"{ranked[1].display_name} is "
        f"{ranked[0].val_metrics['pr_auc'] - ranked[1].val_metrics['pr_auc']:+.3f}."
        if len(ranked) > 1
        else ""
    )
    legit_test = test["tn"] + test["fp"]
    flagged = test["tp"] + test["fp"]

    sections = [
        "# Model Evaluation Report — Credit Card Fraud Detection",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · random seed "
        f"{cfg.training.random_state} · selected model: **{best.display_name}** · "
        f"decision threshold: **{best.threshold:.4f}**",
        "## 1. Data checks before training\n\n"
        f"{verified.report.checks_run} checks passed: schema and values in every split, "
        "counts identical to the Phase 1 report, no row shared between splits, and strict "
        "time order (train → validation → test). The preprocessing inside each model pipeline "
        + (
            "produces exactly the same output as the saved Phase 1 preprocessor."
            if consistent
            else "could not be compared (Phase 1 artifact missing)."
        )
        + "\n\n| Split | Rows | Fraud | Fraud rate |\n|---|---:|---:|---:|\n"
        + split_rows,
        "## 2. Model comparison (validation split)\n\n"
        "All models were trained on the training split only, with class weights instead of "
        "resampling. Each model's threshold was tuned on validation to minimise "
        + (
            f"expected cost (missed fraud = ${th.cost_false_negative:,.0f}, "
            f"false alarm = ${th.cost_false_positive:,.0f})."
            if th.method == "cost"
            else f"F{th.beta:g}."
        )
        + f" Selection metric: validation **{cfg.training.selection_metric.upper()}**.\n\n"
        + rpt.comparison_markdown(table)
        + "\n\n**Why tune the threshold?** 0.5 is an arbitrary cut-off. Each model spreads its "
        "scores differently (class weighting pushes Logistic Regression scores upwards; tree "
        "ensembles are often too cautious), so the best cut-off differs per model and per "
        "cost assumption:\n\n" + rpt.threshold_effect_markdown(table),
        "## 3. Threshold trade-off for the selected model (validation)\n\n"
        + rpt.dataframe_markdown(tradeoff)
        + "\n\nLower thresholds catch more fraud but flag more legitimate customers; the "
        "chosen threshold is the one with the lowest total cost on validation data.",
        "## 4. Final result on the held-out test split (evaluated once)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| PR-AUC | {test['pr_auc']:.3f}{_ci(test_ci, 'pr_auc')} |\n"
        f"| ROC-AUC | {test['roc_auc']:.3f}{_ci(test_ci, 'roc_auc')} |\n"
        f"| Precision | {test['precision']:.3f}{_ci(test_ci, 'precision')} |\n"
        f"| Recall | {test['recall']:.3f}{_ci(test_ci, 'recall')} |\n"
        f"| F1 | {test['f1']:.3f} |\n"
        f"| Confusion matrix | TN={test['tn']:,} · FP={test['fp']} · FN={test['fn']} · "
        f"TP={test['tp']} |\n"
        f"| Expected cost | ${test['expected_cost']:,.0f} vs ${test['cost_without_model']:,.0f} "
        f"with no model ({test['cost_reduction']:.0%} lower) |\n\n"
        "**In plain English:** among the "
        f"{test['n_samples']:,} newest transactions, which the model never saw, there were "
        f"{test['n_positive']} frauds. The model flagged {flagged} transactions: "
        f"{test['tp']} were real fraud and {test['fp']} were false alarms "
        f"({test['fp'] / legit_test:.3%} of the {legit_test:,} legitimate customers). "
        f"It caught {test['recall']:.0%} of the fraud and missed {test['fn']}. "
        f"Roughly {test['precision']:.0%} of alerts are genuine fraud.",
        "## 5. Figures\n\n"
        + "\n".join(f"![{Path(f).stem}](figures/{Path(f).name})" for f in figures),
        "## 6. Caveats\n\n"
        f"- Validation and test each contain only ~{summary['test']['fraud_cases']} frauds, "
        "so metrics move noticeably by chance; read the confidence intervals, not just the "
        f"point values.{gap_sentence}\n"
        f"- Test vs validation PR-AUC: {test['pr_auc'] - best.val_metrics['pr_auc']:+.3f}. "
        "A drop is expected: the winning validation score is "
        "optimistic because it was *chosen* for being highest, and the test period is later "
        "in time. The test numbers are the honest estimate.\n"
        "- The data covers two days. Fraud patterns drift, so performance must be monitored "
        "after deployment (Phase 4).\n"
        "- The costs are illustrative assumptions; change them in `configs/config.yaml` and "
        "re-run to see how the threshold moves.\n"
        "- Hyperparameters are sensible defaults, not tuned. Tuning would use validation "
        "data or time-series cross-validation on the training split, never the test split.",
    ]
    return rpt.write_markdown(out_dir / "evaluation_report.md", sections)
