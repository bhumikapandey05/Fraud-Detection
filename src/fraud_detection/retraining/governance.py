"""Model governance: register candidates, promote with explicit approval, roll back.

Rules enforced here:

* Only ELIGIBLE candidates (all comparison criteria passed) are registered as model
  versions. They start with ``approval_status = pending`` and no alias.
* ``promote`` needs: an eligible, pending version; an unmodified snapshot; a named
  approver; a written reason; and an explicit confirmation (interactive prompt
  unless ``--yes``). It then moves ``champion`` to the new version and
  ``previous-champion`` to the old one.
* ``rollback`` moves ``champion`` back to an APPROVED version (by default the one
  under ``previous-champion``), with the same approver / reason / confirmation.
* Every action is recorded three times: model-version tags, a run in the MLflow
  governance experiment, and the local monitoring store's ``events`` table.
* Aliases never move as a side effect of training or monitoring.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from fraud_detection.config import Config
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import DETECTOR_FILE
from fraud_detection.monitoring.store import MonitoringStore
from fraud_detection.retraining.approval import APPROVED, PENDING, approval_state, is_approved
from fraud_detection.retraining.candidate import verify_snapshot_files
from fraud_detection.tracking.mlflow_tracking import (
    MODEL_ARTIFACT_NAME,
    PACKAGE_DIR,
    configure_mlflow,
)
from fraud_detection.tracking.pyfunc_model import FraudDetectorPyfunc

logger = get_logger(__name__)

PREVIOUS_ALIAS = "previous-champion"


class GovernanceError(RuntimeError):
    """A governance action was refused."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _client(cfg: Config) -> MlflowClient:
    configure_mlflow(cfg)
    return MlflowClient()


def _record(cfg: Config, action: str, from_version, to_version, actor, reason, details) -> None:
    """Governance event -> local store + MLflow governance experiment."""
    MonitoringStore(cfg.paths.monitoring_dir).log_event(
        action, cfg.mlflow.registered_model_name, from_version, to_version, actor, reason, details
    )
    exp_name = cfg.monitoring.governance_experiment_name
    exp = mlflow.get_experiment_by_name(exp_name)
    exp_id = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(
            exp_name, artifact_location=(cfg.mlflow.artifact_dir / "governance").as_uri()
        )
    )
    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"{action}-v{to_version}",
        tags={
            "action": action,
            "from_version": str(from_version),
            "to_version": str(to_version),
            "actor": actor or "",
            "mlflow.note.content": reason or "",
        },
    ):
        mlflow.log_dict(details or {}, "event.json")


# ------------------------------------------------------------------------------------------ #
# Candidates
# ------------------------------------------------------------------------------------------ #
def log_candidate(cfg: Config, candidate_dir: Path) -> dict:
    """Log a candidate run; register it as a pending version only if it is eligible."""
    candidate_dir = Path(candidate_dir)
    verify_snapshot_files(candidate_dir)
    meta = json.loads((candidate_dir / "model" / "model_metadata.json").read_text())
    eligible = meta["criteria"]["eligible"]
    configure_mlflow(cfg)
    tags = {
        "source": "candidate",
        "candidate_id": meta["candidate_id"],
        "candidate_status": "eligible" if eligible else "rejected",
        "champion_version": meta["champion"]["version"],
        "training_fingerprint": meta["training_data"]["training_fingerprint"][:16],
    }
    with mlflow.start_run(run_name=meta["candidate_id"], tags=tags) as run:
        mlflow.set_tag(
            "mlflow.note.content",
            "Candidate retrained on permitted data. "
            + (
                "Eligible for promotion (pending approval)."
                if eligible
                else "Rejected by promotion criteria."
            ),
        )
        mlflow.log_params(
            {
                "random_state": meta["random_state"],
                "threshold.method": meta["threshold"]["method"],
                "training_rows": meta["training_data"]["training_rows"],
                "production_label_rows": meta["training_data"]["sources"]["production_labels"][
                    "rows"
                ],
                "holdout_rows": meta["training_data"]["holdout"]["rows"],
                **{f"model.{k}": v for k, v in (meta["hyperparameters"] or {}).items()},
            }
        )
        metrics = {"threshold": meta["threshold"]["value"]}
        for prefix, block in (
            ("val_", meta["validation_metrics"]),
            ("champion_val_", meta["champion_validation_metrics"]),
            ("holdout_", meta["holdout_metrics"]),
            ("champion_holdout_", meta["champion_holdout_metrics"]),
        ):
            for key in ("pr_auc", "precision", "recall", "f1", "expected_cost"):
                if block:
                    metrics[f"{prefix}{key}"] = float(block[key])
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(candidate_dir / "reports"), "reports")
        mlflow.log_artifact(str(candidate_dir / "data" / "training_manifest.json"), "data")
        mlflow.log_artifact(str(candidate_dir / "MANIFEST.json"), "frozen")
        info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT_NAME,
            python_model=FraudDetectorPyfunc(),
            artifacts={"detector": str(candidate_dir / "model" / DETECTOR_FILE)},
            code_paths=[str(PACKAGE_DIR)],
            metadata={"threshold": meta["threshold"]["value"], "model_name": meta["model_name"]},
        )
    result = {"run_id": run.info.run_id, "eligible": eligible, "version": None}
    if eligible:
        mv = mlflow.register_model(
            info.model_uri,
            cfg.mlflow.registered_model_name,
            tags={
                "model_name": meta["model_name"],
                "threshold": f"{meta['threshold']['value']:.10f}",
                "data_version": meta["training_data"]["training_fingerprint"][:12],
                "source": "candidate",
                "frozen_snapshot": meta["candidate_id"],
                "candidate_status": "eligible",
                "approval_status": PENDING,
                "source_run_id": run.info.run_id,
                "validation_pr_auc": f"{meta['validation_metrics']['pr_auc']:.4f}",
            },
        )
        _client(cfg).update_model_version(
            cfg.mlflow.registered_model_name,
            mv.version,
            description=f"Candidate {meta['candidate_id']} - eligible, awaiting approval.",
        )
        result["version"] = str(mv.version)
        _record(
            cfg,
            "candidate_registered",
            meta["champion"]["version"],
            mv.version,
            "system",
            "passed promotion criteria",
            {"candidate_id": meta["candidate_id"], "checks": meta["criteria"]["checks"]},
        )
    else:
        _record(
            cfg,
            "candidate_rejected",
            meta["champion"]["version"],
            None,
            "system",
            "failed promotion criteria",
            {"candidate_id": meta["candidate_id"], "checks": meta["criteria"]["checks"]},
        )
    return result


# ------------------------------------------------------------------------------------------ #
# Promotion and rollback
# ------------------------------------------------------------------------------------------ #
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GovernanceError(message)


def _champion_version(client: MlflowClient, cfg: Config) -> str:
    return str(
        client.get_model_version_by_alias(
            cfg.mlflow.registered_model_name, cfg.mlflow.champion_alias
        ).version
    )


def promote(
    cfg: Config, version: str, approved_by: str, reason: str, confirm: Callable[[dict], bool]
) -> dict:
    client = _client(cfg)
    name = cfg.mlflow.registered_model_name
    version = str(version)
    _require(bool(approved_by and approved_by.strip()), "--approved-by is required")
    _require(bool(reason and reason.strip()), "--reason is required")
    mv = client.get_model_version(name, version)
    _require(
        mv.tags.get("candidate_status") == "eligible",
        f"v{version} is not an eligible candidate (candidate_status="
        f"{mv.tags.get('candidate_status')!r})",
    )
    _require(
        approval_state(mv.tags) == PENDING,
        f"v{version} is not pending approval (status: {approval_state(mv.tags)})",
    )
    current = _champion_version(client, cfg)
    _require(current != version, f"v{version} is already the champion")
    snapshot = cfg.paths.frozen_dir / mv.tags["frozen_snapshot"]
    verify_snapshot_files(snapshot)
    summary = {
        "action": "promote",
        "from_version": current,
        "to_version": version,
        "snapshot": snapshot.name,
        "approved_by": approved_by,
        "reason": reason,
        "comparison": json.loads((snapshot / "reports" / "comparison.json").read_text()),
    }
    _require(confirm(summary), "Promotion not confirmed - nothing changed")

    stamp = _now()
    for key, value in {
        "approval_status": APPROVED,
        "approved_by": approved_by,
        "approved_at": stamp,
        "approval_reason": reason,
        "promoted_from_version": current,
    }.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_model_version_tag(name, current, "superseded_by", version)
    client.set_registered_model_alias(name, PREVIOUS_ALIAS, current)
    client.set_registered_model_alias(name, cfg.mlflow.champion_alias, version)
    _record(cfg, "promote", current, version, approved_by, reason, summary)
    logger.info(
        "Promoted %s v%s to champion (was v%s), approved by %s", name, version, current, approved_by
    )
    return {
        "champion": version,
        "previous_champion": current,
        "approved_by": approved_by,
        "approved_at": stamp,
    }


def rollback(
    cfg: Config,
    approved_by: str,
    reason: str,
    confirm: Callable[[dict], bool],
    to_version: str | None = None,
) -> dict:
    client = _client(cfg)
    name = cfg.mlflow.registered_model_name
    _require(bool(approved_by and approved_by.strip()), "--approved-by is required")
    _require(bool(reason and reason.strip()), "--reason is required")
    current = _champion_version(client, cfg)
    if to_version is None:
        try:
            to_version = str(client.get_model_version_by_alias(name, PREVIOUS_ALIAS).version)
        except Exception as exc:
            raise GovernanceError("No previous-champion alias; pass --to-version") from exc
    to_version = str(to_version)
    _require(to_version != current, f"v{to_version} is already the champion")
    target = client.get_model_version(name, to_version)
    _require(
        is_approved(target.tags),
        f"v{to_version} was never approved (status: {approval_state(target.tags)}); "
        "only approved versions can be rolled back to",
    )
    snapshot = cfg.paths.frozen_dir / target.tags["frozen_snapshot"]
    _require(snapshot.is_dir(), f"Snapshot {snapshot.name} for v{to_version} is missing")
    verify_snapshot_files(snapshot)
    summary = {
        "action": "rollback",
        "from_version": current,
        "to_version": to_version,
        "snapshot": snapshot.name,
        "approved_by": approved_by,
        "reason": reason,
    }
    _require(confirm(summary), "Rollback not confirmed - nothing changed")

    stamp = _now()
    client.set_model_version_tag(name, current, "rolled_back_at", stamp)
    client.set_model_version_tag(name, current, "rolled_back_by", approved_by)
    client.set_registered_model_alias(name, PREVIOUS_ALIAS, current)
    client.set_registered_model_alias(name, cfg.mlflow.champion_alias, to_version)
    _record(cfg, "rollback", current, to_version, approved_by, reason, summary)
    logger.info(
        "Rolled back %s champion v%s -> v%s (by %s)", name, current, to_version, approved_by
    )
    return {
        "champion": to_version,
        "previous_champion": current,
        "approved_by": approved_by,
        "rolled_back_at": stamp,
    }


def registry_status(cfg: Config) -> dict:
    client = _client(cfg)
    name = cfg.mlflow.registered_model_name
    model = client.get_registered_model(name)
    versions = []
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        aliases = [a for a, v in model.aliases.items() if str(v) == str(mv.version)]
        versions.append(
            {
                "version": str(mv.version),
                "aliases": aliases,
                "approval": approval_state(mv.tags),
                "source": mv.tags.get("source"),
                "snapshot": mv.tags.get("frozen_snapshot"),
                "threshold": mv.tags.get("threshold"),
                "candidate_status": mv.tags.get("candidate_status"),
                "approved_by": mv.tags.get("approved_by"),
                "approved_at": mv.tags.get("approved_at"),
                "validation_pr_auc": mv.tags.get("validation_pr_auc"),
                "test_pr_auc": mv.tags.get("test_pr_auc"),
            }
        )
    events = MonitoringStore(cfg.paths.monitoring_dir).events()
    return {
        "name": name,
        "aliases": {k: str(v) for k, v in model.aliases.items()},
        "versions": versions,
        "events": events.to_dict(orient="records") if isinstance(events, pd.DataFrame) else [],
    }
