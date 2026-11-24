"""Rebuild the project from a fresh clone: data -> model -> frozen release -> champion.

``fraud-detection bootstrap`` runs, in order:

1. download the public Kaggle dataset (public mirror by default; ``--source kaggle``
   uses the Kaggle API) unless ``data/raw/creditcard.csv`` already exists;
2. validate it and prepare the leakage-safe splits;
3. if the frozen release (``phase2-rf-v1``) does NOT exist yet: train the three
   candidate models (Phase 2 pipeline, fixed seeds), then freeze the selected one;
4. verify the frozen release and register it in the local MLflow registry as
   ``fraud-detector@champion`` (idempotent - an existing registration is reused).

Safe to re-run: an existing frozen release is never retrained or replaced, only
verified and registered. Every step is deterministic (fixed seeds), so a fresh clone
reproduces the same splits, model choice, threshold and metrics.
"""

from __future__ import annotations

from pathlib import Path

from fraud_detection.config import Config
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)

RELEASE = "phase2-rf-v1"


def run_bootstrap(
    cfg: Config, config_path: Path, source: str = "mirror", snapshot: str = RELEASE
) -> dict:
    from fraud_detection.data.download import download_dataset
    from fraud_detection.data.loader import load_raw_data
    from fraud_detection.data.validation import validate_raw_data
    from fraud_detection.pipelines.freeze import freeze_model, frozen_path
    from fraud_detection.pipelines.prepare import run_prepare
    from fraud_detection.pipelines.track import track_frozen

    steps: dict[str, str] = {}
    download_dataset(cfg.paths.raw_data.parent, source=source)
    steps["download"] = str(cfg.paths.raw_data)

    report = validate_raw_data(
        load_raw_data(cfg.paths.raw_data), expected_rows=cfg.dataset.expected_rows
    )
    report.log()
    report.raise_if_failed()
    run_prepare(cfg)
    steps["prepare"] = str(cfg.paths.processed_dir)

    if frozen_path(cfg, snapshot).exists():
        logger.info("Frozen release '%s' already exists - verifying, not retraining", snapshot)
        steps["train"] = "skipped (frozen release exists)"
    else:
        from fraud_detection.pipelines.train import run_train

        result = run_train(cfg)
        steps["train"] = f"selected {result['selected_model']}, threshold {result['threshold']:.4f}"
        freeze_model(cfg, snapshot, config_path)
        steps["freeze"] = str(frozen_path(cfg, snapshot))

    tracked = track_frozen(cfg, snapshot, register=True)
    steps["register"] = (
        f"{cfg.mlflow.registered_model_name} v{tracked.registered_version} "
        f"@{cfg.mlflow.champion_alias}"
        + (" (already registered)" if tracked.reused_existing_run else "")
    )
    logger.info("Bootstrap complete: %s", steps)
    return steps
