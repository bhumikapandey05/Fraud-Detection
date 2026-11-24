"""Build the explainer for the served model, with the same fail-safe spirit as serving.

The SHAP background must come from the exact training split the frozen model was
trained on: its SHA-256 is compared with the frozen snapshot's manifest. If that
check (or anything else) fails, explanations are disabled - predictions are not
affected.
"""

from __future__ import annotations

import json

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.explainability.shap_explainer import (
    ExplanationError,
    FraudExplainer,
    sample_background,
)
from fraud_detection.logging_utils import get_logger
from fraud_detection.pipelines.train import _file_sha256

logger = get_logger(__name__)


def build_explainer(cfg: Config, detector, snapshot: str) -> FraudExplainer:
    train_path = cfg.paths.processed_dir / "train.parquet"
    if not train_path.is_file():
        raise ExplanationError(f"Training split not found at {train_path}")
    manifest = json.loads((cfg.paths.frozen_dir / snapshot / "MANIFEST.json").read_text())
    train_sha = _file_sha256(train_path)
    if train_sha != manifest["split_files_sha256"]["train"]:
        raise ExplanationError(
            "Training split differs from the one the frozen model was "
            "trained on; refusing to build a SHAP background from it"
        )
    ecfg = cfg.explainability
    background = sample_background(
        pd.read_parquet(train_path), ecfg.background_size, ecfg.random_state
    )
    explainer = FraudExplainer(
        detector,
        background,
        additivity_tolerance=ecfg.additivity_tolerance,
        background_info={
            "source": "training split",
            "random_state": ecfg.random_state,
            "train_split_sha256": train_sha,
        },
    )
    logger.info(
        "SHAP explainer ready (background %d training rows, baseline %.5f)",
        ecfg.background_size,
        explainer.base_value,
    )
    return explainer
