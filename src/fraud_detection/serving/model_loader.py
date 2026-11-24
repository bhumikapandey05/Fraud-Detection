"""Load the APPROVED model for serving - and refuse anything else.

The service never trains, tunes or picks a model. It loads the release that was
approved in Phase 3 and checks, before accepting traffic, that what it loaded
is byte-for-byte that release:

* ``registry`` source (default): resolve ``fraud-detector@champion`` in the local
  MLflow registry, require the version to be tagged with the expected frozen
  snapshot, and compare the SHA-256 of its ``fraud_detector.joblib`` and its
  threshold with the frozen snapshot's ``MANIFEST.json``. With
  ``serving.frozen_snapshot: auto`` the expected snapshot is the one the champion
  version is tagged with, and the version must additionally be APPROVED
  (see ``retraining.approval``) - so promotions and rollbacks take effect on the
  next start, while unapproved versions are refused.
* ``frozen`` source: load the frozen snapshot folder directly (after its own
  checksum verification) - useful when MLflow is not available.

If any check fails the service starts but reports *not ready* (HTTP 503) and
rejects predictions: failing closed is safer than serving an unknown model.
No data split is read here, so loading never touches test data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import DETECTOR_FILE, FraudDetector, load_detector
from fraud_detection.pipelines.train import _file_sha256

logger = get_logger(__name__)

MANIFEST_FILE = "MANIFEST.json"
DETECTOR_MANIFEST_KEY = f"model/{DETECTOR_FILE}"


class ModelLoadError(RuntimeError):
    """The approved model could not be loaded or failed verification."""


@dataclass
class ServedModel:
    """Everything the API needs to score transactions and describe the model."""

    detector: FraudDetector
    threshold: float
    model_name: str
    display_name: str
    source: str
    snapshot: str
    registered_model: str | None = None
    version: str | None = None
    alias: str | None = None
    run_id: str | None = None
    data_version: str | None = None
    detector_sha256: str = ""
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Fraud probability per row (same object used in Phase 2 evaluation)."""
        return pd.Series(
            self.detector.predict_proba(features[FEATURE_COLUMNS]), index=features.index
        )

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "threshold": self.threshold,
            "source": self.source,
            "frozen_snapshot": self.snapshot,
            "registered_model": self.registered_model,
            "version": self.version,
            "alias": self.alias,
            "run_id": self.run_id,
            "data_version": self.data_version,
            "detector_sha256": self.detector_sha256,
            "loaded_at": self.loaded_at,
        }


def _load_manifest(cfg: Config, snapshot: str) -> tuple[Path, dict]:
    root = cfg.paths.frozen_dir / snapshot
    path = root / MANIFEST_FILE
    if not path.is_file():
        raise ModelLoadError(f"Frozen snapshot '{snapshot}' not found at {root}")
    manifest = json.loads(path.read_text())
    if DETECTOR_MANIFEST_KEY not in manifest.get("files", {}):
        raise ModelLoadError(f"Manifest of '{snapshot}' does not list {DETECTOR_MANIFEST_KEY}")
    return root, manifest


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ModelLoadError(message)


def load_from_frozen(cfg: Config, snapshot: str) -> ServedModel:
    if snapshot == "auto":
        raise ModelLoadError("serving.model_source 'frozen' needs a snapshot name, not 'auto'")
    root, manifest = _load_manifest(cfg, snapshot)
    detector_path = root / DETECTOR_MANIFEST_KEY
    sha = _file_sha256(detector_path)
    _check(
        sha == manifest["files"][DETECTOR_MANIFEST_KEY],
        f"Detector file in snapshot '{snapshot}' does not match its manifest checksum",
    )
    meta = json.loads((root / "model" / "model_metadata.json").read_text())
    detector = load_detector(root / "model")
    _check(
        math.isclose(detector.threshold, manifest["threshold"], abs_tol=1e-12),
        "Detector threshold differs from the frozen manifest",
    )
    return ServedModel(
        detector=detector,
        threshold=detector.threshold,
        model_name=meta["model_name"],
        display_name=meta["display_name"],
        source="frozen",
        snapshot=snapshot,
        data_version=None,
        detector_sha256=sha,
    )


def load_from_registry(cfg: Config, snapshot: str) -> ServedModel:
    if cfg.mlflow is None:
        raise ModelLoadError("serving.model_source is 'registry' but no 'mlflow' config exists")
    from mlflow.tracking import MlflowClient

    from fraud_detection.retraining.approval import approval_state, is_approved
    from fraud_detection.tracking.mlflow_tracking import configure_mlflow, mlflow

    if snapshot != "auto":
        _load_manifest(cfg, snapshot)  # fail fast on a missing pinned snapshot
    mcfg = cfg.mlflow
    if not mcfg.tracking_db.is_file():
        raise ModelLoadError(f"MLflow database not found at {mcfg.tracking_db}")
    configure_mlflow(cfg)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(mcfg.registered_model_name, mcfg.champion_alias)
    except Exception as exc:
        raise ModelLoadError(
            f"No '{mcfg.champion_alias}' alias on registered model '{mcfg.registered_model_name}'"
        ) from exc

    if snapshot == "auto":
        snapshot = mv.tags.get("frozen_snapshot") or ""
        _check(bool(snapshot), f"Champion v{mv.version} has no 'frozen_snapshot' tag")
        _check(
            is_approved(mv.tags),
            f"Champion v{mv.version} is not approved (status: {approval_state(mv.tags)}); "
            "promote it with `fraud-detection promote` or roll back",
        )
    _, manifest = _load_manifest(cfg, snapshot)
    _check(
        mv.tags.get("frozen_snapshot") == snapshot,
        f"{mcfg.registered_model_name}@{mcfg.champion_alias} (v{mv.version}) is not the "
        f"approved snapshot '{snapshot}' (tag={mv.tags.get('frozen_snapshot')!r})",
    )

    model_uri = f"models:/{mcfg.registered_model_name}@{mcfg.champion_alias}"
    local_dir = Path(mlflow.artifacts.download_artifacts(model_uri))
    detector_path = local_dir / "artifacts" / DETECTOR_FILE
    _check(detector_path.is_file(), f"Registered model has no artifacts/{DETECTOR_FILE}")
    sha = _file_sha256(detector_path)
    _check(
        sha == manifest["files"][DETECTOR_MANIFEST_KEY],
        "Registered detector differs from the frozen snapshot (SHA-256 mismatch)",
    )

    pyfunc = mlflow.pyfunc.load_model(model_uri)
    detector = pyfunc.unwrap_python_model().detector
    _check(isinstance(detector, FraudDetector), "Registered model does not wrap a FraudDetector")
    _check(
        math.isclose(detector.threshold, manifest["threshold"], abs_tol=1e-12),
        "Registered threshold differs from the frozen snapshot",
    )
    _check(
        math.isclose(float(mv.tags.get("threshold", "nan")), detector.threshold, abs_tol=1e-9),
        "Registry 'threshold' tag differs from the model's threshold",
    )

    return ServedModel(
        detector=detector,
        threshold=detector.threshold,
        model_name=mv.tags.get("model_name", detector.model_name),
        display_name=detector.model_name.replace("_", " ").title(),
        source="registry",
        snapshot=snapshot,
        registered_model=mcfg.registered_model_name,
        version=str(mv.version),
        alias=mcfg.champion_alias,
        run_id=mv.tags.get("source_run_id"),
        data_version=mv.tags.get("data_version"),
        detector_sha256=sha,
    )


def load_served_model(cfg: Config) -> ServedModel:
    scfg = cfg.serving
    loader = load_from_registry if scfg.model_source == "registry" else load_from_frozen
    model = loader(cfg, scfg.frozen_snapshot)
    logger.info(
        "Serving %s (source=%s, version=%s, threshold=%.4f, sha256=%s...)",
        model.display_name,
        model.source,
        model.version,
        model.threshold,
        model.detector_sha256[:12],
    )
    return model
