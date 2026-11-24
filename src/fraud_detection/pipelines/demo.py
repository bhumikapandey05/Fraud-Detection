"""End-to-end monitoring + retraining demo in an isolated SANDBOX.

Everything happens under ``--workdir`` (default ``artifacts/demo``): its own MLflow
registry, monitoring store, frozen snapshots and reports. The real registry, the
real champion and the real MLflow history are never touched. The sandbox starts
from a copy of the frozen Phase 2 release.

Story (one synthetic batch per "day", all generated from the TRAINING split):

1. day 1 stable, day 2 amount shift, day 3 feature shift, day 4 data-quality issues,
   days 5-6 concept drift -> ingest, attach labels, monitor (alerts expected on 2-6)
2. retrain a candidate on permitted data, compare with the champion
3. if eligible: promote with an explicit (demo) approval, monitor day 7 with it
4. roll back to the original champion with an explicit (demo) approval
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml

from fraud_detection.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, load_config
from fraud_detection.logging_utils import get_logger
from fraud_detection.monitoring.simulate import generate_batch
from fraud_detection.pipelines.monitor import add_labels, ingest_batch, run_monitoring
from fraud_detection.serving.model_loader import load_served_model

logger = get_logger(__name__)

DEMO_DAYS = [
    (1, "stable"),
    (2, "amount_shift"),
    (3, "feature_shift"),
    (4, "quality_issues"),
    (5, "concept_drift"),
    (6, "concept_drift"),
]
DEMO_APPROVER = "demo-reviewer (simulated approval in sandbox)"


def setup_sandbox(
    workdir: Path,
    base: Config | None = None,
    snapshot: str = "phase2-rf-v1",
    base_config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[Path, Config]:
    base = base or load_config(base_config_path)
    workdir = Path(workdir).resolve()
    if workdir.exists():
        for p in workdir.rglob("*"):  # frozen snapshots are read-only
            if p.is_file():
                p.chmod(0o644)
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    raw = yaml.safe_load(Path(base_config_path).read_text())
    raw["paths"] = {
        k: str(Path(v) if Path(v).is_absolute() else PROJECT_ROOT / v)
        for k, v in raw["paths"].items()
    }
    # Read-only inputs stay where they are; everything written goes into the sandbox.
    raw["paths"]["processed_dir"] = str(base.paths.processed_dir)
    raw["paths"]["raw_data"] = str(base.paths.raw_data)
    for key in (
        "artifacts_dir",
        "model_dir",
        "training_report_dir",
        "frozen_dir",
        "monitoring_dir",
        "eda_dir",
        "log_dir",
        "interim_dir",
    ):
        raw["paths"][key] = str(workdir / key.replace("_dir", ""))
    raw["mlflow"]["tracking_db"] = str(workdir / "mlflow" / "mlflow.db")
    raw["mlflow"]["artifact_dir"] = str(workdir / "mlflow" / "artifacts")
    raw["serving"]["frozen_snapshot"] = "auto"
    config_path = workdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(config_path)

    shutil.copytree(base.paths.frozen_dir / snapshot, cfg.paths.frozen_dir / snapshot)
    _register_champion(cfg, snapshot)
    return config_path, cfg


def _register_champion(cfg: Config, snapshot: str) -> None:
    """Register the copied release as v1@champion (same tags as `track-frozen`)."""
    from fraud_detection.tracking.mlflow_tracking import log_training_run

    root = cfg.paths.frozen_dir / snapshot
    log_training_run(
        cfg,
        model_dir=root / "model",
        report_dir=root / "reports",
        prep_report=root / "data" / "data_preparation_report.json",
        config_path=root / "config" / "config.yaml",
        run_name=f"{snapshot} (frozen, sandbox)",
        tags={"source": "frozen_snapshot", "frozen_snapshot": snapshot},
        register=True,
        alias=cfg.mlflow.champion_alias,
    )


def _day(cfg: Config, train: pd.DataFrame, day: int, scenario: str, n_rows: int, seed: int) -> dict:
    batch_id = f"day{day}-{scenario}"
    x, y = generate_batch(train, scenario, n_rows, seed=seed + day, day=day, batch_id=batch_id)
    incoming = cfg.paths.monitoring_dir / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    x.to_csv(incoming / f"{batch_id}.csv", index=False)
    y.to_csv(incoming / f"{batch_id}_labels.csv", index=False)
    ingest_batch(cfg, incoming / f"{batch_id}.csv", batch_id, model=load_served_model(cfg))
    add_labels(cfg, incoming / f"{batch_id}_labels.csv")
    return run_monitoring(cfg, batch_id)


def run_demo(workdir: Path, n_rows: int = 20000, seed: int = 100) -> dict:
    from fraud_detection.retraining.candidate import train_candidate
    from fraud_detection.retraining.governance import log_candidate, promote, registry_status
    from fraud_detection.retraining.governance import rollback as do_rollback

    config_path, cfg = setup_sandbox(workdir)
    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    runs = [_day(cfg, train, day, scenario, n_rows, seed) for day, scenario in DEMO_DAYS]

    candidate = train_candidate(cfg)
    logged = log_candidate(cfg, Path(candidate["snapshot_dir"]))
    promoted = rolled_back = None
    if logged["eligible"]:
        promoted = promote(
            cfg,
            logged["version"],
            DEMO_APPROVER,
            "Demo: candidate recovers recall under concept drift",
            confirm=lambda summary: True,
        )
        runs.append(_day(cfg, train, 7, "concept_drift", n_rows, seed))
        rolled_back = do_rollback(
            cfg,
            DEMO_APPROVER,
            "Demo: rollback drill to the Phase 2 release",
            confirm=lambda summary: True,
        )
    return {
        "workdir": str(Path(workdir).resolve()),
        "config": str(config_path),
        "runs": runs,
        "candidate": candidate,
        "candidate_logged": logged,
        "promoted": promoted,
        "rolled_back": rolled_back,
        "registry": registry_status(cfg),
    }


def sandbox_config(workdir: Path) -> Config:
    return load_config(Path(workdir) / "config.yaml")


__all__ = ["run_demo", "setup_sandbox", "sandbox_config", "replace"]
