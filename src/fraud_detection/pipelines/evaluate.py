"""Re-evaluate the SAVED detector on the held-out test split.

This does not select or tune anything: it loads the exact artifact that would
be deployed, scores the test split, and confirms the numbers match the ones
recorded at training time. A mismatch means the artifact, the data or the
environment changed - i.e. the result is not reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

from fraud_detection.config import Config
from fraud_detection.data.verification import load_splits
from fraud_detection.features.preprocessing import split_features_target
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import load_detector, load_metadata
from fraud_detection.models.metrics import classification_metrics

logger = get_logger(__name__)

COMPARED_METRICS = ("pr_auc", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn")


def run_evaluate(cfg: Config, model_dir: Path | None = None) -> dict:
    model_dir = Path(model_dir) if model_dir else cfg.paths.model_dir
    detector = load_detector(model_dir)
    metadata = load_metadata(model_dir)
    logger.info("Loaded %r", detector)

    test = load_splits(cfg.paths.processed_dir).test
    X_test, y_test = split_features_target(test, cfg.dataset.target)
    metrics = classification_metrics(
        y_test,
        detector.predict_proba(X_test),
        detector.threshold,
        cost_false_positive=cfg.threshold.cost_false_positive,
        cost_false_negative=cfg.threshold.cost_false_negative,
    )

    recorded = metadata.get("test_metrics", {})
    mismatches = {
        k: {"recorded": recorded.get(k), "now": metrics[k]}
        for k in COMPARED_METRICS
        if k not in recorded or not math.isclose(recorded[k], metrics[k], abs_tol=1e-9)
    }
    if mismatches:
        logger.warning("Test metrics differ from those recorded at training: %s", mismatches)
    else:
        logger.info("Reproduced recorded test metrics exactly")
    logger.info(
        "TEST  PR-AUC=%.4f ROC-AUC=%.4f precision=%.3f recall=%.3f (TP=%d FP=%d FN=%d)",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["precision"],
        metrics["recall"],
        metrics["tp"],
        metrics["fp"],
        metrics["fn"],
    )
    return {
        "model": metadata["display_name"],
        "threshold": detector.threshold,
        "test_metrics": metrics,
        "reproducible": not mismatches,
        "mismatches": mismatches,
    }
