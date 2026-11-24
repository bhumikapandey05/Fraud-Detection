"""MLflow tracking pipelines.

* ``track_frozen``   - log a frozen snapshot (e.g. the Phase 2 model) to MLflow and
                       register it, WITHOUT retraining. Idempotent: running it again
                       reuses the existing run instead of creating duplicates.
* ``track_training`` - log the result of the latest ``fraud-detection train``.
                       New versions are only registered on request and never take
                       over the champion alias automatically.
* ``registry_summary`` - list registered versions and aliases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fraud_detection.config import Config
from fraud_detection.logging_utils import get_logger
from fraud_detection.pipelines.freeze import (
    MANIFEST_FILE,
    PREP_REPORT_FILE,
    frozen_path,
    verify_frozen,
)
from fraud_detection.pipelines.train import _file_sha256
from fraud_detection.tracking.mlflow_tracking import (
    MlflowClient,
    TrackingResult,
    configure_mlflow,
    find_runs_by_tag,
    log_training_run,
    require_mlflow_config,
)

logger = get_logger(__name__)


def _registered_version_for_run(cfg: Config, run_id: str) -> str | None:
    mcfg = require_mlflow_config(cfg)
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{mcfg.registered_model_name}'")
    except Exception:  # registered model does not exist yet
        return None
    for mv in versions:
        if mv.tags.get("source_run_id") == run_id:
            return str(mv.version)
    return None


def track_frozen(cfg: Config, name: str, register: bool = True) -> TrackingResult:
    """Log + register a frozen snapshot; the snapshot is verified first."""
    verify_frozen(cfg, name)
    mcfg = require_mlflow_config(cfg)
    root = frozen_path(cfg, name)

    existing = find_runs_by_tag(cfg, "frozen_snapshot", name)
    if existing:
        run = existing[0]
        version = _registered_version_for_run(cfg, run.info.run_id)
        logger.info(
            "Snapshot '%s' is already tracked in run %s; not logging it again",
            name,
            run.info.run_id,
        )
        if register and version is None:
            raise RuntimeError(
                f"Run {run.info.run_id} exists but is not registered; register it from the "
                "MLflow UI or delete the run and re-run track-frozen."
            )
        return TrackingResult(run.info.run_id, run.info.experiment_id, None, version, True)

    manifest = json.loads((root / MANIFEST_FILE).read_text())
    return log_training_run(
        cfg,
        model_dir=root / "model",
        report_dir=root / "reports",
        prep_report=root / "data" / PREP_REPORT_FILE,
        config_path=root / "config" / "config.yaml",
        run_name=f"{name} (frozen)",
        tags={
            "source": "frozen_snapshot",
            "frozen_snapshot": name,
            "frozen_at": manifest["frozen_at"],
            "frozen_manifest_sha256": _file_sha256(root / MANIFEST_FILE),
            "raw_data_sha256": str(manifest.get("raw_data_sha256")),
        },
        register=register,
        alias=mcfg.champion_alias if register else None,
        extra_artifacts={"frozen": root / MANIFEST_FILE},
    )


def track_training(cfg: Config, config_path: Path, register: bool = False) -> TrackingResult:
    """Log the working output of the latest ``train`` run (never moves the champion alias)."""
    return log_training_run(
        cfg,
        model_dir=cfg.paths.model_dir,
        report_dir=cfg.paths.training_report_dir,
        prep_report=cfg.paths.artifacts_dir / PREP_REPORT_FILE,
        config_path=config_path,
        run_name=f"train-{datetime.now(UTC):%Y%m%d-%H%M%S}",
        tags={"source": "train"},
        register=register,
        alias=None,
    )


def registry_summary(cfg: Config) -> dict:
    """Registered versions of the model, newest first, with their aliases."""
    mcfg = require_mlflow_config(cfg)
    configure_mlflow(cfg)
    client = MlflowClient()
    try:
        model = client.get_registered_model(mcfg.registered_model_name)
    except Exception:
        return {"name": mcfg.registered_model_name, "aliases": {}, "versions": []}
    versions = client.search_model_versions(f"name='{mcfg.registered_model_name}'")
    return {
        "name": model.name,
        "aliases": dict(model.aliases),
        "versions": [
            {
                "version": mv.version,
                "model_name": mv.tags.get("model_name"),
                "threshold": mv.tags.get("threshold"),
                "test_pr_auc": mv.tags.get("test_pr_auc"),
                "data_version": mv.tags.get("data_version"),
                "source": mv.tags.get("source"),
                "source_run_id": mv.tags.get("source_run_id"),
            }
            for mv in sorted(versions, key=lambda v: int(v.version), reverse=True)
        ],
    }
