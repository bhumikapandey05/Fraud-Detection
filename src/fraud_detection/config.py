"""Load and validate the project configuration (``configs/config.yaml``).

The YAML file is parsed into small, typed dataclasses so that the rest of the
code base gets attribute access (``cfg.split.train_size``) and early, clear
errors when a value is wrong, instead of failing deep inside a pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigError(ValueError):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class PathsConfig:
    raw_data: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    eda_dir: Path
    log_dir: Path
    model_dir: Path
    training_report_dir: Path
    frozen_dir: Path
    # Added in Phase 5. Has a default so configs frozen with earlier releases still load.
    monitoring_dir: Path = PROJECT_ROOT / "monitoring"


@dataclass(frozen=True)
class DatasetConfig:
    kaggle_slug: str = "mlg-ulb/creditcardfraud"
    file_name: str = "creditcard.csv"
    target: str = "Class"
    time_column: str = "Time"
    amount_column: str = "Amount"
    expected_rows: int | None = 284807


@dataclass(frozen=True)
class CleaningConfig:
    drop_duplicates: bool = True


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "time"
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    min_positives_per_split: int = 30
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.strategy not in {"time", "stratified"}:
            raise ConfigError(
                f"split.strategy must be 'time' or 'stratified', got {self.strategy!r}"
            )
        sizes = (self.train_size, self.validation_size, self.test_size)
        if any(not 0 < s < 1 for s in sizes):
            raise ConfigError(f"split sizes must each be in (0, 1), got {sizes}")
        if not math.isclose(sum(sizes), 1.0, abs_tol=1e-6):
            raise ConfigError(f"split sizes must sum to 1.0, got {sum(sizes):.4f}")
        if self.min_positives_per_split < 1:
            raise ConfigError("split.min_positives_per_split must be >= 1")


@dataclass(frozen=True)
class PreprocessingConfig:
    amount_transform: str = "log1p"
    scaler: str = "robust"
    add_hour_features: bool = True
    drop_raw_time: bool = True

    def __post_init__(self) -> None:
        if self.amount_transform not in {"log1p", "none"}:
            raise ConfigError("preprocessing.amount_transform must be 'log1p' or 'none'")
        if self.scaler not in {"robust", "standard"}:
            raise ConfigError("preprocessing.scaler must be 'robust' or 'standard'")


SUPPORTED_MODELS = ("logistic_regression", "random_forest", "xgboost")
SELECTION_METRICS = ("pr_auc", "roc_auc")


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    models: tuple[str, ...] = SUPPORTED_MODELS
    selection_metric: str = "pr_auc"
    bootstrap_samples: int = 1000
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        unknown = [m for m in self.models if m not in SUPPORTED_MODELS]
        if not self.models or unknown:
            raise ConfigError(
                f"training.models must be a non-empty subset of {SUPPORTED_MODELS}, "
                f"got unknown {unknown}"
            )
        unknown_params = [m for m in self.params if m not in SUPPORTED_MODELS]
        if unknown_params:
            raise ConfigError(f"training.params has unknown models: {unknown_params}")
        if self.selection_metric not in SELECTION_METRICS:
            raise ConfigError(f"training.selection_metric must be one of {SELECTION_METRICS}")
        if self.bootstrap_samples < 0:
            raise ConfigError("training.bootstrap_samples must be >= 0")


@dataclass(frozen=True)
class ThresholdConfig:
    method: str = "cost"
    cost_false_positive: float = 5.0
    cost_false_negative: float = 120.0
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.method not in {"cost", "f_beta"}:
            raise ConfigError("threshold.method must be 'cost' or 'f_beta'")
        if self.cost_false_positive <= 0 or self.cost_false_negative <= 0:
            raise ConfigError("threshold costs must be positive")
        if self.beta <= 0:
            raise ConfigError("threshold.beta must be positive")


@dataclass(frozen=True)
class MlflowConfig:
    tracking_db: Path
    artifact_dir: Path
    experiment_name: str = "credit-card-fraud-detection"
    registered_model_name: str = "fraud-detector"
    champion_alias: str = "champion"
    log_training_runs: bool = True

    @property
    def tracking_uri(self) -> str:
        """SQLite URI for the local tracking + model registry database."""
        return f"sqlite:///{self.tracking_db}"


@dataclass(frozen=True)
class ServingConfig:
    model_source: str = "registry"
    frozen_snapshot: str = "phase2-rf-v1"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_batch_size: int = 1000
    dashboard_api_url: str = "http://127.0.0.1:8000"

    def __post_init__(self) -> None:
        if self.model_source not in {"registry", "frozen"}:
            raise ConfigError("serving.model_source must be 'registry' or 'frozen'")
        if not 1 <= self.max_batch_size <= 100_000:
            raise ConfigError("serving.max_batch_size must be between 1 and 100000")
        if not 1 <= self.api_port <= 65535:
            raise ConfigError("serving.api_port must be a valid port")


@dataclass(frozen=True)
class ExplainabilityConfig:
    background_size: int = 500
    random_state: int = 42
    top_k: int = 5
    additivity_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if not 10 <= self.background_size <= 5000:
            raise ConfigError("explainability.background_size must be between 10 and 5000")
        if not 1 <= self.top_k <= 30:
            raise ConfigError("explainability.top_k must be between 1 and 30")
        if not 0 < self.additivity_tolerance < 1e-2:
            raise ConfigError("explainability.additivity_tolerance must be in (0, 0.01)")


@dataclass(frozen=True)
class AlertThresholds:
    drift_share_warning: float = 0.20
    drift_share_critical: float = 0.40
    watched_features: tuple[str, ...] = ("Amount", "hour_of_day", "V14", "V17", "V12", "V10")
    alert_rate_ratio_max: float = 3.0
    missing_share_max: float = 0.0
    out_of_range_share_max: float = 0.01
    duplicate_share_max: float = 0.01
    min_recall: float = 0.65
    min_precision: float = 0.25
    min_pr_auc: float = 0.60

    def __post_init__(self) -> None:
        if not 0 < self.drift_share_warning <= self.drift_share_critical <= 1:
            raise ConfigError("monitoring.alerts: need 0 < drift_share_warning <= critical <= 1")
        object.__setattr__(self, "watched_features", tuple(self.watched_features))
        if self.alert_rate_ratio_max <= 1:
            raise ConfigError("monitoring.alerts.alert_rate_ratio_max must be > 1")


@dataclass(frozen=True)
class MonitoringConfig:
    capture_predictions: bool = True
    evidently_python: str = ".venv-monitoring/bin/python"
    experiment_name: str = "credit-card-fraud-monitoring"
    governance_experiment_name: str = "credit-card-fraud-governance"
    reference_size: int = 20000
    random_state: int = 42
    min_rows: int = 200
    min_fraud_labels: int = 20
    alerts: AlertThresholds = field(default_factory=AlertThresholds)

    def __post_init__(self) -> None:
        if isinstance(self.alerts, dict):
            object.__setattr__(
                self, "alerts", _build(AlertThresholds, self.alerts, "monitoring.alerts")
            )
        if self.reference_size < 100 or self.min_rows < 10 or self.min_fraud_labels < 1:
            raise ConfigError("monitoring sizes are too small")

    def evidently_python_path(self) -> Path:
        path = Path(self.evidently_python)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class PromotionCriteria:
    max_validation_pr_auc_drop: float = 0.01
    max_validation_recall_drop: float = 0.05
    max_validation_cost_increase: float = 0.10
    min_holdout_pr_auc_gain: float = 0.0
    require_holdout: bool = True


@dataclass(frozen=True)
class RetrainingConfig:
    use_production_labels: bool = True
    holdout_fraction: float = 0.30
    random_state: int = 42
    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)

    def __post_init__(self) -> None:
        if isinstance(self.criteria, dict):
            object.__setattr__(
                self, "criteria", _build(PromotionCriteria, self.criteria, "retraining.criteria")
            )
        if not 0 < self.holdout_fraction < 1:
            raise ConfigError("retraining.holdout_fraction must be in (0, 1)")


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    mlflow: MlflowConfig | None = None
    serving: ServingConfig = field(default_factory=ServingConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build(section_cls: type, values: dict[str, Any] | None, section: str):
    """Instantiate a dataclass section, turning unknown keys into a clear error."""
    values = values or {}
    try:
        return section_cls(**values)
    except TypeError as exc:  # unknown / missing keys
        raise ConfigError(f"Invalid '{section}' section in config: {exc}") from exc


def _build_mlflow(values: dict[str, Any] | None, base: Path) -> MlflowConfig | None:
    """MLflow settings are optional; storage paths are resolved like ``paths``."""
    if not values:
        return None
    values = dict(values)
    for key in ("tracking_db", "artifact_dir"):
        if key not in values:
            raise ConfigError(f"mlflow.{key} is required when an 'mlflow' section is present")
        values[key] = (base / values[key]).resolve()
    return _build(MlflowConfig, values, "mlflow")


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """Read the YAML config and return a validated :class:`Config`.

    Args:
        path: Path to a YAML file. Defaults to ``configs/config.yaml``.
        root: Directory that relative paths are resolved against.
            Defaults to the project root.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level")

    base = Path(root) if root else PROJECT_ROOT
    raw_paths = raw.get("paths")
    if not raw_paths:
        raise ConfigError("Config is missing the required 'paths' section")
    resolved = {key: (base / value).resolve() for key, value in raw_paths.items()}

    return Config(
        paths=_build(PathsConfig, resolved, "paths"),
        dataset=_build(DatasetConfig, raw.get("dataset"), "dataset"),
        cleaning=_build(CleaningConfig, raw.get("cleaning"), "cleaning"),
        split=_build(SplitConfig, raw.get("split"), "split"),
        preprocessing=_build(PreprocessingConfig, raw.get("preprocessing"), "preprocessing"),
        training=_build(TrainingConfig, raw.get("training"), "training"),
        threshold=_build(ThresholdConfig, raw.get("threshold"), "threshold"),
        mlflow=_build_mlflow(raw.get("mlflow"), base),
        serving=_build(ServingConfig, raw.get("serving"), "serving"),
        explainability=_build(ExplainabilityConfig, raw.get("explainability"), "explainability"),
        monitoring=_build(MonitoringConfig, raw.get("monitoring"), "monitoring"),
        retraining=_build(RetrainingConfig, raw.get("retraining"), "retraining"),
        logging=_build(LoggingConfig, raw.get("logging"), "logging"),
    )
