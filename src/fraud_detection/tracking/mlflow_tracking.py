"""Log training results to a local MLflow store and register model versions.

Everything is read from the artifacts a training run already saved (model
metadata, reports, config), so the same code logs both:

* a brand-new ``fraud-detection train`` run, and
* an existing frozen snapshot (e.g. the Phase 2 model) - without retraining.

Layout of one logged training run in MLflow::

    parent run  "<run name>"         selected model, threshold, val + test metrics,
    │                                data version, config, reports, registered model
    ├── child run "Logistic Regression"   params + validation metrics
    ├── child run "Random Forest"         params + validation metrics
    └── child run "XGBoost"               params + validation metrics
"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from fraud_detection.config import PROJECT_ROOT, Config, MlflowConfig, load_config  # noqa: E402
from fraud_detection.data.verification import SPLIT_NAMES  # noqa: E402
from fraud_detection.logging_utils import get_logger  # noqa: E402
from fraud_detection.models.detector import DETECTOR_FILE, load_metadata  # noqa: E402
from fraud_detection.models.factory import MODEL_DISPLAY_NAMES  # noqa: E402
from fraud_detection.pipelines.train import _file_sha256  # noqa: E402
from fraud_detection.tracking.pyfunc_model import FraudDetectorPyfunc  # noqa: E402

logger = get_logger(__name__)
logging.getLogger("mlflow").setLevel(logging.WARNING)

MODEL_ARTIFACT_NAME = "fraud_detector"
PACKAGE_DIR = PROJECT_ROOT / "src" / "fraud_detection"
LOGGED_METRICS = (
    "pr_auc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
    "alert_rate",
    "expected_cost",
    "cost_reduction",
)


class TrackingError(RuntimeError):
    """Raised when MLflow logging cannot be done safely."""


@dataclass(frozen=True)
class TrackingResult:
    run_id: str
    experiment_id: str
    model_uri: str | None
    registered_version: str | None
    reused_existing_run: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------- #
# Setup
# ------------------------------------------------------------------------------------- #
def require_mlflow_config(cfg: Config) -> MlflowConfig:
    if cfg.mlflow is None:
        raise TrackingError("No 'mlflow' section in the config")
    return cfg.mlflow


def configure_mlflow(cfg: Config) -> str:
    """Point MLflow at the local SQLite store; create the experiment if needed."""
    mcfg = require_mlflow_config(cfg)
    mcfg.tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mcfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mcfg.tracking_uri)
    mlflow.set_registry_uri(mcfg.tracking_uri)
    experiment = mlflow.get_experiment_by_name(mcfg.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            mcfg.experiment_name, artifact_location=mcfg.artifact_dir.as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def flatten(values: dict, prefix: str = "") -> dict[str, str]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": "1"}`` (MLflow params are flat strings)."""
    out: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = str(value)
    return out


def data_version(split_hashes: dict[str, str]) -> str:
    """Short, stable ID for the exact train/validation/test files used."""
    joined = "|".join(f"{k}:{split_hashes[k]}" for k in sorted(split_hashes))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def _metrics(prefix: str, metrics: dict) -> dict[str, float]:
    return {f"{prefix}{k}": float(metrics[k]) for k in LOGGED_METRICS if k in metrics}


def _ci_metrics(prefix: str, ci: dict) -> dict[str, float]:
    out = {}
    for key, (low, high) in (ci or {}).items():
        out[f"{prefix}{key}_ci_low"] = float(low)
        out[f"{prefix}{key}_ci_high"] = float(high)
    return out


# ------------------------------------------------------------------------------------- #
# Logging
# ------------------------------------------------------------------------------------- #
def find_runs_by_tag(cfg: Config, key: str, value: str) -> list:
    experiment_id = configure_mlflow(cfg)
    return MlflowClient().search_runs(
        [experiment_id], filter_string=f"tags.`{key}` = '{value}'", max_results=10
    )


def log_training_run(
    cfg: Config,
    *,
    model_dir: Path,
    report_dir: Path,
    prep_report: Path,
    config_path: Path,
    run_name: str,
    tags: dict[str, str] | None = None,
    register: bool = False,
    alias: str | None = None,
    extra_artifacts: dict[str, Path] | None = None,
) -> TrackingResult:
    """Log one saved training result (all candidates + selected model) to MLflow."""
    mcfg = require_mlflow_config(cfg)
    experiment_id = configure_mlflow(cfg)
    meta = load_metadata(model_dir)
    run_cfg = load_config(config_path)  # the configuration that produced this model

    # The data on disk must be exactly what the model was trained/evaluated on.
    split_hashes = meta["data"]["files_sha256"]
    on_disk = {s: _file_sha256(cfg.paths.processed_dir / f"{s}.parquet") for s in SPLIT_NAMES}
    if on_disk != split_hashes:
        raise TrackingError(
            "Split files on disk differ from those recorded in the model metadata; refusing to "
            "log a run with a misleading data version."
        )
    version = data_version(split_hashes)
    threshold = meta["threshold"]

    common_params = {
        "random_state": run_cfg.training.random_state,
        "selection_metric": run_cfg.training.selection_metric,
        **flatten(asdict(run_cfg.split), "split."),
        **flatten(asdict(run_cfg.preprocessing), "preprocessing."),
        "cleaning.drop_duplicates": run_cfg.cleaning.drop_duplicates,
        **flatten({k: v for k, v in threshold.items() if k != "value"}, "threshold."),
        "imbalance_strategy": meta["class_imbalance"]["strategy"],
        "data_version": version,
    }
    all_tags = {
        "project": "credit-card-fraud-detection",
        "selected_model": meta["model_name"],
        "data_version": version,
        **{f"data.{s}_sha256": h for s, h in split_hashes.items()},
        "model_trained_at": meta["created_at"],
        **(tags or {}),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MLflow schema/pip hints are noise here
        with mlflow.start_run(run_name=run_name, tags=all_tags) as parent:
            mlflow.set_tag(
                "mlflow.note.content",
                f"{meta['display_name']} selected by validation "
                f"{meta['selection']['metric']}; threshold {threshold['value']:.4f} tuned on "
                f"validation ({threshold['method']}). Test evaluated once.",
            )
            mlflow.log_params(common_params)
            mlflow.log_params(flatten(meta["hyperparameters"], "model."))
            mlflow.log_metric("threshold", threshold["value"])
            mlflow.log_metrics(_metrics("val_", meta["validation_metrics"]))
            mlflow.log_metrics(_ci_metrics("val_", meta.get("validation_ci95")))
            mlflow.log_metrics(_metrics("test_", meta["test_metrics"]))
            mlflow.log_metrics(_ci_metrics("test_", meta.get("test_ci95")))

            # Training data version: MLflow dataset entries (name + digest + source).
            for split in SPLIT_NAMES:
                path = cfg.paths.processed_dir / f"{split}.parquet"
                dataset = mlflow.data.from_pandas(
                    pd.read_parquet(path),
                    source=str(path),
                    name=f"{split}-{version}",
                    targets=cfg.dataset.target,
                )
                mlflow.log_input(dataset, context=split)

            # Files: reports, metadata, data-prep report, config.
            mlflow.log_artifacts(str(report_dir), artifact_path="reports")
            mlflow.log_artifact(str(Path(model_dir) / "model_metadata.json"), "model_metadata")
            mlflow.log_artifact(str(prep_report), "data")
            mlflow.log_artifact(str(config_path), "config")
            for artifact_path, file in (extra_artifacts or {}).items():
                mlflow.log_artifact(str(file), artifact_path)

            # One nested run per candidate, for side-by-side comparison in the UI.
            for name, scores in meta["selection"]["candidates"].items():
                with mlflow.start_run(
                    run_name=MODEL_DISPLAY_NAMES.get(name, name),
                    nested=True,
                    tags={
                        "candidate": name,
                        "selected": str(name == meta["model_name"]),
                        "data_version": version,
                    },
                ):
                    mlflow.log_params(
                        {
                            "model_name": name,
                            **flatten(run_cfg.training.params.get(name, {}), "model."),
                            "random_state": run_cfg.training.random_state,
                            "data_version": version,
                        }
                    )
                    mlflow.log_metrics({f"val_{k}": float(v) for k, v in scores.items()})

            # The deployable model: pipeline + threshold, self-contained with our code.
            model_info = mlflow.pyfunc.log_model(
                name=MODEL_ARTIFACT_NAME,
                python_model=FraudDetectorPyfunc(),
                artifacts={"detector": str(Path(model_dir) / DETECTOR_FILE)},
                code_paths=[str(PACKAGE_DIR)],
                input_example=pd.read_parquet(cfg.paths.processed_dir / "validation.parquet").head(
                    5
                )[meta["input_features"]],
                pip_requirements=_pip_requirements(meta),
                metadata={"threshold": threshold["value"], "model_name": meta["model_name"]},
            )

    registered_version = None
    if register:
        registered_version = register_model(
            cfg, model_info.model_uri, meta, version, parent.info.run_id, all_tags, alias
        )
    logger.info(
        "Logged MLflow run %s (%s) in experiment '%s'",
        parent.info.run_id,
        run_name,
        mcfg.experiment_name,
    )
    return TrackingResult(
        parent.info.run_id, experiment_id, model_info.model_uri, registered_version
    )


def _pip_requirements(meta: dict) -> list[str]:
    env = meta["environment"]
    return [
        f"scikit-learn=={env['scikit-learn']}",
        f"pandas=={env['pandas']}",
        f"numpy=={env['numpy']}",
        f"xgboost=={env['xgboost']}",
        "joblib",
        "scipy",
        "pyyaml",
    ]


def register_model(
    cfg: Config,
    model_uri: str,
    meta: dict,
    version: str,
    run_id: str,
    tags: dict[str, str],
    alias: str | None,
) -> str:
    """Create a new version in the Model Registry (optionally pointing an alias at it)."""
    mcfg = require_mlflow_config(cfg)
    client = MlflowClient()
    mv = mlflow.register_model(
        model_uri,
        mcfg.registered_model_name,
        tags={
            "model_name": meta["model_name"],
            "threshold": f"{meta['threshold']['value']:.10f}",
            "data_version": version,
            "test_pr_auc": f"{meta['test_metrics']['pr_auc']:.4f}",
            "source_run_id": run_id,
            **{k: v for k, v in tags.items() if k in ("source", "frozen_snapshot")},
        },
    )
    client.update_model_version(
        mcfg.registered_model_name,
        mv.version,
        description=(
            f"{meta['display_name']} with decision threshold {meta['threshold']['value']:.4f} "
            f"(tuned on validation). Test PR-AUC {meta['test_metrics']['pr_auc']:.3f}, recall "
            f"{meta['test_metrics']['recall']:.3f}, precision "
            f"{meta['test_metrics']['precision']:.3f}. Data version {version}."
        ),
    )
    if alias:
        client.set_registered_model_alias(mcfg.registered_model_name, alias, mv.version)
        logger.info("Alias '%s' -> %s version %s", alias, mcfg.registered_model_name, mv.version)
    logger.info("Registered %s version %s", mcfg.registered_model_name, mv.version)
    return str(mv.version)


def load_registered_model(cfg: Config, alias: str | None = None, version: str | None = None):
    """Load a registered model as an MLflow pyfunc (``predict`` returns score + decision)."""
    mcfg = require_mlflow_config(cfg)
    configure_mlflow(cfg)
    ref = f"@{alias or mcfg.champion_alias}" if version is None else f"/{version}"
    return mlflow.pyfunc.load_model(f"models:/{mcfg.registered_model_name}{ref}")
