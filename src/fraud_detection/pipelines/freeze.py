"""Freeze a trained model as an immutable, verifiable snapshot.

``artifacts/model`` and ``reports/training`` are *working* outputs: every
``fraud-detection train`` overwrites them and ``make clean`` deletes them. A
released model needs to outlive both. Freezing copies everything that defines
the release into ``artifacts/frozen/<name>/``:

    model/fraud_detector.joblib       fitted preprocessing + model + threshold
    model/model_metadata.json         metrics, threshold, data hashes, versions
    reports/...                       evaluation report, tables, figures
    reports/validation_vs_test.*      documented validation -> test gap
    data/data_preparation_report.json how the splits were produced
    config/config.yaml                configuration used
    MANIFEST.json                     SHA-256 of every file above

The files are made read-only, and ``verify_frozen`` re-checks the checksums,
confirms the split files are byte-identical to the ones the model was trained
and evaluated on, and re-scores the test split to reproduce the recorded metrics.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

from fraud_detection.analysis.generalization import (
    compare_validation_test,
    write_generalization_report,
)
from fraud_detection.config import Config
from fraud_detection.data.verification import SPLIT_NAMES, load_splits
from fraud_detection.features.preprocessing import split_features_target
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import (
    DETECTOR_FILE,
    METADATA_FILE,
    load_detector,
    load_metadata,
)
from fraud_detection.models.metrics import classification_metrics
from fraud_detection.pipelines.evaluate import COMPARED_METRICS
from fraud_detection.pipelines.train import _file_sha256

logger = get_logger(__name__)

MANIFEST_FILE = "MANIFEST.json"
PREP_REPORT_FILE = "data_preparation_report.json"


class FrozenModelError(RuntimeError):
    """Raised when a frozen snapshot is missing, altered or not reproducible."""


def frozen_path(cfg: Config, name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"Invalid snapshot name {name!r}")
    return cfg.paths.frozen_dir / name


def run_gap_report(cfg: Config, model_dir: Path, out_dir: Path) -> dict:
    """Validation-vs-test analysis for the detector saved in ``model_dir`` (read-only)."""
    detector = load_detector(model_dir)
    meta = load_metadata(model_dir)
    splits = load_splits(cfg.paths.processed_dir)
    X_val, y_val = split_features_target(splits.validation, cfg.dataset.target)
    X_test, y_test = split_features_target(splits.test, cfg.dataset.target)
    th = meta["threshold"]
    result = compare_validation_test(
        y_val,
        detector.predict_proba(X_val),
        y_test,
        detector.predict_proba(X_test),
        detector.threshold,
        cost_false_positive=th["cost_false_positive"],
        cost_false_negative=th["cost_false_negative"],
        random_state=meta["random_state"],
    )
    paths = write_generalization_report(result, meta["display_name"], out_dir)
    logger.info("Validation-vs-test report written to %s", paths["markdown"])
    return result


def _make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _manifest_files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): _file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST_FILE
    }


def freeze_model(cfg: Config, name: str, config_path: Path) -> Path:
    """Snapshot the current working model + reports under ``frozen_dir/name``."""
    dest = frozen_path(cfg, name)
    if dest.exists():
        raise FileExistsError(
            f"Snapshot '{name}' already exists at {dest}; frozen models are "
            "never overwritten. Choose a new name."
        )

    # Only freeze a model that still reproduces its own recorded test metrics.
    _reproduce_test_metrics(cfg, cfg.paths.model_dir)

    tmp = dest.with_name(f".{name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        (tmp / "model").mkdir(parents=True)
        for file in (DETECTOR_FILE, METADATA_FILE):
            shutil.copy2(cfg.paths.model_dir / file, tmp / "model" / file)
        shutil.copytree(cfg.paths.training_report_dir, tmp / "reports")
        (tmp / "data").mkdir()
        shutil.copy2(cfg.paths.artifacts_dir / PREP_REPORT_FILE, tmp / "data" / PREP_REPORT_FILE)
        (tmp / "config").mkdir()
        shutil.copy2(config_path, tmp / "config" / "config.yaml")
        run_gap_report(cfg, tmp / "model", tmp / "reports")

        meta = load_metadata(tmp / "model")
        manifest = {
            "name": name,
            "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model_name": meta["model_name"],
            "threshold": meta["threshold"]["value"],
            "trained_at": meta["created_at"],
            "split_files_sha256": meta["data"]["files_sha256"],
            "raw_data_sha256": _file_sha256(cfg.paths.raw_data)
            if cfg.paths.raw_data.is_file()
            else None,
            "files": _manifest_files(tmp),
        }
        (tmp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
        _make_read_only(tmp)
        os.replace(tmp, dest)  # appears atomically, only when complete
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info("Frozen snapshot '%s' written to %s", name, dest)
    return dest


def _reproduce_test_metrics(cfg: Config, model_dir: Path) -> dict:
    detector = load_detector(model_dir)
    meta = load_metadata(model_dir)
    test = load_splits(cfg.paths.processed_dir).test
    X, y = split_features_target(test, cfg.dataset.target)
    metrics = classification_metrics(y, detector.predict_proba(X), detector.threshold)
    recorded = meta["test_metrics"]
    bad = [k for k in COMPARED_METRICS if not math.isclose(recorded[k], metrics[k], abs_tol=1e-9)]
    if bad:
        raise FrozenModelError(f"Test metrics not reproduced for {model_dir}: {bad}")
    return metrics


def load_manifest(cfg: Config, name: str) -> dict:
    path = frozen_path(cfg, name) / MANIFEST_FILE
    if not path.is_file():
        raise FrozenModelError(
            f"Snapshot '{name}' not found (no {path}). Run `fraud-detection "
            f"freeze --name {name}` first."
        )
    return json.loads(path.read_text())


def verify_frozen(cfg: Config, name: str) -> dict:
    """Check integrity + data identity + reproducibility of a snapshot. Raises on failure."""
    root = frozen_path(cfg, name)
    manifest = load_manifest(cfg, name)

    current = _manifest_files(root)
    changed = sorted(k for k, v in manifest["files"].items() if current.get(k) != v)
    added = sorted(set(current) - set(manifest["files"]))
    if changed or added:
        raise FrozenModelError(f"Snapshot '{name}' was modified: changed={changed} added={added}")

    split_hashes = {s: _file_sha256(cfg.paths.processed_dir / f"{s}.parquet") for s in SPLIT_NAMES}
    if split_hashes != manifest["split_files_sha256"]:
        raise FrozenModelError(
            "The split files differ from the ones this model was trained/evaluated on. "
            "Re-run `fraud-detection prepare` with the frozen config to restore them."
        )

    metrics = _reproduce_test_metrics(cfg, root / "model")
    logger.info(
        "Snapshot '%s' verified: %d files intact, data identical, test metrics "
        "reproduced (PR-AUC=%.4f)",
        name,
        len(current),
        metrics["pr_auc"],
    )
    return {"name": name, "files_checked": len(current), "test_metrics": metrics, "ok": True}
