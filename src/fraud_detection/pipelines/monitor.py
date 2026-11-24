"""Monitoring pipelines: ingest batches, attach labels, run monitoring.

``run_monitoring`` for one batch of served predictions:

1. load the predictions (features + served scores) and any labels;
2. load the reference for the model version that served them;
3. data quality (pandas), feature + score drift (Evidently, isolated env),
   review-rate check, labelled performance;
4. evaluate alert rules;
5. save HTML reports + summary, store the run and alerts, log to MLflow.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from fraud_detection.config import Config
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.logging_utils import get_logger
from fraud_detection.monitoring.alerts import evaluate_alerts, overall_status
from fraud_detection.monitoring.drift import run_drift
from fraud_detection.monitoring.performance import labeled_performance
from fraud_detection.monitoring.quality import data_quality
from fraud_detection.monitoring.reference import build_reference
from fraud_detection.monitoring.store import MonitoringStore
from fraud_detection.serving.model_loader import load_served_model
from fraud_detection.serving.schemas import Transaction

logger = get_logger(__name__)


def get_store(cfg: Config) -> MonitoringStore:
    return MonitoringStore(cfg.paths.monitoring_dir)


# ------------------------------------------------------------------------------------------ #
# Ingestion and labels
# ------------------------------------------------------------------------------------------ #
def ingest_batch(cfg: Config, file: Path, batch_id: str | None = None, model=None) -> dict:
    """Validate a CSV of transactions like the API does, score valid rows, store them."""
    file = Path(file)
    batch_id = batch_id or file.stem
    frame = pd.read_csv(file)
    model = model or load_served_model(cfg)

    valid_rows, reasons = [], {}
    records = frame.to_dict(orient="records")
    for i, rec in enumerate(records):
        payload = {k: rec[k] for k in FEATURE_COLUMNS if k in rec}
        payload = {k: (None if pd.isna(v) else v) for k, v in payload.items()}
        try:
            Transaction(**{k: float(v) if v is not None else None for k, v in payload.items()})
            valid_rows.append(i)
        except (ValidationError, TypeError, ValueError) as exc:
            errors = exc.errors() if isinstance(exc, ValidationError) else [{"type": "invalid"}]
            for err in errors:
                reasons[err["type"]] = reasons.get(err["type"], 0) + 1

    valid = frame.iloc[valid_rows].reset_index(drop=True)
    ids = (
        valid["transaction_id"].astype(str)
        if "transaction_id" in valid
        else pd.Series([f"{batch_id}-{i}" for i in valid_rows])
    )
    store = get_store(cfg)
    if len(valid):
        scores = model.score(valid[FEATURE_COLUMNS].astype(float))
        store.log_predictions(
            valid,
            ids.tolist(),
            scores.to_numpy(),
            model.threshold,
            model.version,
            batch_id,
            source="batch-file",
        )
    store.record_ingestion(
        batch_id,
        str(file),
        len(frame),
        len(valid),
        {"n_rejected": len(frame) - len(valid), "reasons": reasons},
    )
    result = {
        "batch_id": batch_id,
        "rows": len(frame),
        "scored": len(valid),
        "rejected": len(frame) - len(valid),
        "rejection_reasons": reasons,
        "model_version": model.version,
    }
    logger.info(
        "Ingested %s: %d scored, %d rejected", batch_id, len(valid), len(frame) - len(valid)
    )
    return result


def add_labels(cfg: Config, file: Path) -> dict:
    labels = pd.read_csv(file)
    result = get_store(cfg).add_labels(labels, source=Path(file).name)
    logger.info("Labels: %s", result)
    return result


# ------------------------------------------------------------------------------------------ #
# Monitoring run
# ------------------------------------------------------------------------------------------ #
def _version_detector(cfg: Config, version: str | None):
    """Detector + metadata of the version that served the predictions."""
    from fraud_detection.models.detector import load_detector, load_metadata

    snapshot = None
    if cfg.mlflow is not None and version is not None:
        from mlflow.tracking import MlflowClient

        from fraud_detection.tracking.mlflow_tracking import configure_mlflow

        configure_mlflow(cfg)
        mv = MlflowClient().get_model_version(cfg.mlflow.registered_model_name, version)
        snapshot = mv.tags.get("frozen_snapshot")
    if snapshot is None:
        snapshot = cfg.serving.frozen_snapshot
    root = cfg.paths.frozen_dir / snapshot / "model"
    from fraud_detection.pipelines.train import _file_sha256

    return (
        load_detector(root),
        load_metadata(root),
        snapshot,
        _file_sha256(root / "fraud_detector.joblib"),
    )


def run_monitoring(cfg: Config, batch_id: str | None = None, log_to_mlflow: bool = True) -> dict:
    store = get_store(cfg)
    batches = store.batches()
    if batches.empty:
        raise ValueError("The monitoring store has no predictions yet")
    batch_id = batch_id or batches.iloc[-1]["batch_id"]
    window = store.predictions(batch_id=batch_id)
    if window.empty:
        raise ValueError(f"No predictions for batch '{batch_id}'")

    versions = window["model_version"].fillna("unknown").value_counts()
    version = versions.index[0]
    detector, meta, snapshot, sha = _version_detector(
        cfg, None if version == "unknown" else version
    )
    reference = build_reference(cfg, detector, version, sha)
    threshold = float(window["threshold"].iloc[0])

    created = datetime.now(UTC)
    run_id = f"{created:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    report_dir = Path(cfg.paths.monitoring_dir) / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    features = window[FEATURE_COLUMNS]
    ingestion = store.ingestion(batch_id)
    quality = data_quality(
        features,
        reference.train_low,
        reference.train_high,
        rejected=ingestion["rejections"] if ingestion else None,
    )
    drift = run_drift(
        cfg,
        reference.features,
        features,
        reference.validation_scores,
        window["fraud_score"],
        report_dir,
    )
    alert_rate = float(window["is_fraud"].mean())
    scores = {
        "mean": float(window["fraud_score"].mean()),
        "median": float(window["fraud_score"].median()),
        "p99": float(window["fraud_score"].quantile(0.99)),
        "alert_rate": alert_rate,
        "expected_alert_rate": reference.expected_alert_rate,
        "alert_rate_ratio": alert_rate / reference.expected_alert_rate
        if reference.expected_alert_rate
        else None,
        "reference_mean": float(reference.validation_scores.mean()),
    }
    th = meta["threshold"]
    performance = labeled_performance(
        window,
        threshold,
        th["cost_false_positive"],
        th["cost_false_negative"],
        cfg.monitoring.min_fraud_labels,
    )
    summary = {
        "run_id": run_id,
        "created_at": created.isoformat(timespec="seconds"),
        "batch_id": batch_id,
        "n_rows": len(window),
        "model_version": version,
        "model_versions_in_window": versions.to_dict(),
        "snapshot": snapshot,
        "threshold": threshold,
        "reference": {k: reference.info["key"][k] for k in ("reference_size", "random_state")}
        | {"features": "training split sample", "scores": "validation split"},
        "quality": quality,
        "drift": drift,
        "scores": scores,
        "performance": performance,
        "report_dir": str(report_dir),
    }
    alerts = evaluate_alerts(summary, cfg.monitoring.alerts, cfg.monitoring.min_rows)
    summary["status"] = overall_status(alerts)
    summary["alerts"] = alerts
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if log_to_mlflow and cfg.mlflow is not None:
        summary["mlflow_run_id"] = _log_monitoring_run(cfg, summary, report_dir)
    store.save_run(summary, alerts)
    logger.info(
        "Monitoring %s (%s): status=%s, %d alerts, drift share %.0f%%",
        batch_id,
        run_id,
        summary["status"],
        len(alerts),
        100 * drift["drift_share"],
    )
    return summary


def _log_monitoring_run(cfg: Config, summary: dict, report_dir: Path) -> str:
    import mlflow

    from fraud_detection.tracking.mlflow_tracking import configure_mlflow

    configure_mlflow(cfg)
    exp = mlflow.get_experiment_by_name(cfg.monitoring.experiment_name)
    exp_id = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(
            cfg.monitoring.experiment_name,
            artifact_location=(cfg.mlflow.artifact_dir / "monitoring").as_uri(),
        )
    )
    perf, q, sc, d = (
        summary["performance"],
        summary["quality"],
        summary["scores"],
        summary["drift"],
    )
    metrics = {
        "n_rows": summary["n_rows"],
        "drift_share": d["drift_share"],
        "n_drifted": d["n_drifted"],
        "score_drift": d["score_drift"]["value"],
        "score_mean": sc["mean"],
        "alert_rate": sc["alert_rate"],
        "rejected_share": q["rejected_share"],
        "out_of_range_share": q["out_of_range_share"],
        "duplicate_share": q["duplicate_share"],
        "n_alerts": len(summary["alerts"]),
        "label_coverage": perf["label_coverage"],
    }
    if sc["alert_rate_ratio"] is not None:
        metrics["alert_rate_ratio"] = sc["alert_rate_ratio"]
    if perf["status"] == "evaluated":
        metrics |= {k: perf[k] for k in ("pr_auc", "precision", "recall", "f1", "expected_cost")}
    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"monitor-{summary['batch_id']}",
        tags={
            "batch_id": summary["batch_id"],
            "status": summary["status"],
            "model_version": str(summary["model_version"]),
            "snapshot": summary["snapshot"],
        },
    ) as run:
        mlflow.log_params(
            {
                "batch_id": summary["batch_id"],
                "threshold": summary["threshold"],
                "reference_size": summary["reference"]["reference_size"],
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifacts(str(report_dir), artifact_path="monitoring")
    return run.info.run_id
