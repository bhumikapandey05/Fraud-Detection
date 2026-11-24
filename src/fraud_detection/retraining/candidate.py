"""Train a reproducible candidate and compare it with the champion.

The candidate reuses the champion's recipe exactly - same preprocessing config,
same Random Forest hyperparameters, same cost-based threshold rule - read from the
champion's frozen ``config/config.yaml``. Only the training data changes. The
threshold is tuned on the validation split, as in Phase 2. The test split is never
read (see ``retraining.data``).

Output: a read-only candidate snapshot ``<frozen_dir>/candidate-<id>/`` with the
detector, metadata, comparison report and a checksum manifest.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn

from fraud_detection.config import Config, load_config
from fraud_detection.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import FraudDetector, load_detector, save_detector
from fraud_detection.models.factory import build_training_pipeline
from fraud_detection.models.metrics import classification_metrics
from fraud_detection.models.thresholds import tune_threshold
from fraud_detection.monitoring.store import MonitoringStore
from fraud_detection.pipelines.freeze import _make_read_only, _manifest_files
from fraud_detection.retraining.compare import evaluate_criteria
from fraud_detection.retraining.data import assemble_training_data

logger = get_logger(__name__)

MODEL_NAME = "random_forest"


@dataclass
class Champion:
    version: str
    snapshot: str
    root: Path
    detector: FraudDetector
    manifest: dict
    config: Config


def load_champion(cfg: Config) -> Champion:
    from mlflow.tracking import MlflowClient

    from fraud_detection.tracking.mlflow_tracking import configure_mlflow

    configure_mlflow(cfg)
    mv = MlflowClient().get_model_version_by_alias(
        cfg.mlflow.registered_model_name, cfg.mlflow.champion_alias
    )
    snapshot = mv.tags["frozen_snapshot"]
    root = cfg.paths.frozen_dir / snapshot
    return Champion(
        version=str(mv.version),
        snapshot=snapshot,
        root=root,
        detector=load_detector(root / "model"),
        manifest=json.loads((root / "MANIFEST.json").read_text()),
        config=load_config(root / "config" / "config.yaml"),
    )


def _metrics(y, scores, threshold, th_cfg) -> dict:
    return classification_metrics(
        y,
        scores,
        threshold,
        cost_false_positive=th_cfg.cost_false_positive,
        cost_false_negative=th_cfg.cost_false_negative,
    )


def train_candidate(cfg: Config, champion: Champion | None = None) -> dict:
    champion = champion or load_champion(cfg)
    recipe = champion.config  # the champion's own training configuration
    rcfg = cfg.retraining
    data = assemble_training_data(
        cfg.paths.processed_dir,
        MonitoringStore(cfg.paths.monitoring_dir),
        champion.manifest,
        rcfg.use_production_labels,
        rcfg.holdout_fraction,
    )
    X, y = data.train[FEATURE_COLUMNS], data.train[TARGET_COLUMN]
    pipeline = build_training_pipeline(
        MODEL_NAME,
        recipe.training.params.get(MODEL_NAME),
        rcfg.random_state,
        recipe.preprocessing,
        y_train=y,
    )
    logger.info("Training candidate on %d rows (%d frauds)", len(X), int(y.sum()))
    pipeline.fit(X, y)

    Xv, yv = data.validation[FEATURE_COLUMNS], data.validation[TARGET_COLUMN]
    cand_val_scores = pipeline.predict_proba(Xv)[:, 1]
    threshold = tune_threshold(yv, cand_val_scores, recipe.threshold).threshold
    candidate = FraudDetector(pipeline, threshold, MODEL_NAME, FEATURE_COLUMNS)

    th = recipe.threshold
    cand_val = _metrics(yv, cand_val_scores, threshold, th)
    champ_val = _metrics(yv, champion.detector.predict_proba(Xv), champion.detector.threshold, th)

    cand_hold = champ_hold = None
    hold = data.holdout
    if (
        len(hold)
        and int(hold[TARGET_COLUMN].sum()) >= cfg.monitoring.min_fraud_labels
        and hold[TARGET_COLUMN].nunique() == 2
    ):
        Xh, yh = hold[FEATURE_COLUMNS], hold[TARGET_COLUMN]
        cand_hold = _metrics(yh, candidate.predict_proba(Xh), threshold, th)
        champ_hold = _metrics(
            yh, champion.detector.predict_proba(Xh), champion.detector.threshold, th
        )
    decision = evaluate_criteria(cand_val, champ_val, cand_hold, champ_hold, rcfg.criteria)

    created = datetime.now(UTC)
    # Timestamp + data fingerprint + random suffix: unique even for identical reruns.
    candidate_id = (
        f"candidate-{created:%Y%m%d-%H%M%S}-"
        f"{data.manifest['training_fingerprint'][:8]}-{uuid.uuid4().hex[:4]}"
    )
    metadata = {
        "model_name": MODEL_NAME,
        "display_name": "Random Forest",
        "kind": "candidate",
        "candidate_id": candidate_id,
        "created_at": created.isoformat(timespec="seconds"),
        "random_state": rcfg.random_state,
        "champion": {"version": champion.version, "snapshot": champion.snapshot},
        "threshold": {
            "value": threshold,
            "tuned_on": "validation",
            "method": th.method,
            "cost_false_positive": th.cost_false_positive,
            "cost_false_negative": th.cost_false_negative,
            "beta": th.beta,
        },
        "hyperparameters": recipe.training.params.get(MODEL_NAME),
        "validation_metrics": cand_val,
        "holdout_metrics": cand_hold,
        "champion_validation_metrics": champ_val,
        "champion_holdout_metrics": champ_hold,
        "criteria": decision,
        "input_features": FEATURE_COLUMNS,
        "training_data": data.manifest,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    root = _write_snapshot(cfg, candidate_id, candidate, metadata, champion)
    logger.info(
        "Candidate %s: %s",
        candidate_id,
        "ELIGIBLE for promotion" if decision["eligible"] else "NOT eligible",
    )
    return {"candidate_id": candidate_id, "snapshot_dir": str(root), "metadata": metadata}


def _metric_row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['pr_auc']:.4f} | {m['precision']:.3f} | {m['recall']:.3f} | "
        f"{m['threshold']:.4f} | {m['fp']} | {m['fn']} | ${m['expected_cost']:,.0f} |"
    )


def _comparison_markdown(meta: dict) -> str:
    rows = "\n".join(
        f"| {c['description']} | {_fmt(c['candidate'])} | {_fmt(c['champion'])} | "
        f"{'pass' if c['passed'] else 'FAIL'} |"
        for c in meta["criteria"]["checks"]
    )
    data = meta["training_data"]
    prod = data["sources"]["production_labels"]
    verdict = (
        "**ELIGIBLE** - may be promoted after explicit human approval."
        if meta["criteria"]["eligible"]
        else "**NOT ELIGIBLE** - cannot be promoted."
    )
    holdout_rows = ""
    if meta["holdout_metrics"]:
        holdout_rows = (
            "\n\n## Recent labelled holdout (not trained on)\n\n"
            + _TABLE_HEAD
            + _metric_row("Candidate", meta["holdout_metrics"])
            + "\n"
            + _metric_row("Champion", meta["champion_holdout_metrics"])
        )
    return f"""# Candidate {meta["candidate_id"]}

Champion: v{meta["champion"]["version"]} ({meta["champion"]["snapshot"]}). Same preprocessing,
hyperparameters and threshold rule as the champion; only the training data differs.

## Verdict

{verdict}

| Criterion | Candidate | Champion | Result |
|---|---:|---:|---|
{rows}

## Validation split (shared yardstick)

{_TABLE_HEAD}{_metric_row("Candidate", meta["validation_metrics"])}
{_metric_row("Champion", meta["champion_validation_metrics"])}{holdout_rows}

## Training data

* Training split: {data["sources"]["train_split"]["rows"]:,} rows
  ({data["sources"]["train_split"]["frauds"]} frauds)
* Labelled production rows: {prod["rows"]:,} ({prod["frauds"]} frauds) from
  {", ".join(prod["batches"]) or "none"}
* Recent holdout (not trained on): {data["holdout"]["rows"]:,} rows
  ({data["holdout"]["frauds"]} frauds)
* Test split: {data["test_split"]}
* Training fingerprint: `{data["training_fingerprint"][:16]}`
"""


_TABLE_HEAD = (
    "| | PR-AUC | Precision | Recall | Threshold | False alarms | Missed | Cost |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|\n"
)


def _fmt(v) -> str:
    return "—" if v is None else f"{v:,.4f}" if abs(v) < 100 else f"{v:,.0f}"


def _write_snapshot(
    cfg: Config, candidate_id: str, detector: FraudDetector, metadata: dict, champion: Champion
) -> Path:
    dest = cfg.paths.frozen_dir / candidate_id
    tmp = dest.with_name(f".{candidate_id}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        save_detector(detector, metadata, tmp / "model")
        (tmp / "reports").mkdir(parents=True)
        (tmp / "reports" / "comparison.md").write_text(_comparison_markdown(metadata))
        (tmp / "reports" / "comparison.json").write_text(json.dumps(metadata["criteria"], indent=2))
        (tmp / "data").mkdir()
        (tmp / "data" / "training_manifest.json").write_text(
            json.dumps(metadata["training_data"], indent=2)
        )
        (tmp / "config").mkdir()
        shutil.copy2(champion.root / "config" / "config.yaml", tmp / "config" / "config.yaml")
        manifest = {
            "name": candidate_id,
            "kind": "candidate",
            "frozen_at": metadata["created_at"],
            "model_name": MODEL_NAME,
            "threshold": detector.threshold,
            "champion_version": champion.version,
            "eligible": metadata["criteria"]["eligible"],
            # Train/validation hashes let serving + explanations verify their inputs.
            "split_files_sha256": {
                "train": metadata["training_data"]["sources"]["train_split"]["sha256"],
                "validation": metadata["training_data"]["validation_sha256"],
            },
            "files": _manifest_files(tmp),
        }
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        _make_read_only(tmp)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def verify_snapshot_files(root: Path) -> None:
    """Checksums of any snapshot (champion or candidate) still match its manifest."""
    manifest = json.loads((Path(root) / "MANIFEST.json").read_text())
    current = _manifest_files(Path(root))
    bad = sorted(k for k, v in manifest["files"].items() if current.get(k) != v)
    if bad or set(current) - set(manifest["files"]):
        raise RuntimeError(f"Snapshot {Path(root).name} was modified: {bad}")
