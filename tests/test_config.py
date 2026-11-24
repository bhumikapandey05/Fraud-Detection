from pathlib import Path

import pytest
import yaml

from fraud_detection.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ExplainabilityConfig,
    PreprocessingConfig,
    ServingConfig,
    SplitConfig,
    ThresholdConfig,
    TrainingConfig,
    load_config,
)


def test_default_config_loads():
    cfg = load_config()
    assert cfg.dataset.target == "Class"
    assert cfg.split.strategy == "time"
    assert cfg.paths.raw_data.is_absolute()


def test_relative_paths_resolve_against_root(tmp_path: Path):
    cfg = load_config(DEFAULT_CONFIG_PATH, root=tmp_path)
    assert cfg.paths.raw_data == (tmp_path / "data/raw/creditcard.csv").resolve()


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_unknown_key_raises(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["split"]["typo_key"] = 1
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="split"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_size": 0.8, "validation_size": 0.15, "test_size": 0.15},  # sums to 1.1
        {"train_size": 1.0, "validation_size": 0.0, "test_size": 0.0},
        {"strategy": "random"},
        {"min_positives_per_split": 0},
    ],
)
def test_invalid_split_config(kwargs):
    with pytest.raises(ConfigError):
        SplitConfig(**kwargs)


def test_invalid_preprocessing_config():
    with pytest.raises(ConfigError):
        PreprocessingConfig(scaler="minmax")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"models": ["svm"]},
        {"models": []},
        {"selection_metric": "accuracy"},
        {"bootstrap_samples": -1},
        {"params": {"svm": {}}},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ConfigError):
        TrainingConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "magic"}, {"cost_false_positive": 0}, {"cost_false_negative": -1}, {"beta": 0}],
)
def test_invalid_threshold_config(kwargs):
    with pytest.raises(ConfigError):
        ThresholdConfig(**kwargs)


def test_default_config_has_training_sections():
    cfg = load_config()
    assert cfg.training.models == ("logistic_regression", "random_forest", "xgboost")
    assert cfg.threshold.method == "cost"
    assert cfg.paths.model_dir.name == "model"


def test_mlflow_config_resolves_paths(tmp_path):
    cfg = load_config(DEFAULT_CONFIG_PATH, root=tmp_path)
    assert cfg.mlflow.tracking_db == (tmp_path / "mlflow/mlflow.db").resolve()
    assert cfg.mlflow.tracking_uri == f"sqlite:///{cfg.mlflow.tracking_db}"
    assert cfg.mlflow.champion_alias == "champion"


def test_mlflow_section_is_optional_but_validated(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    no_mlflow = {k: v for k, v in raw.items() if k != "mlflow"}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(no_mlflow))
    assert load_config(path).mlflow is None
    raw["mlflow"].pop("tracking_db")
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="tracking_db"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs", [{"model_source": "s3"}, {"max_batch_size": 0}, {"api_port": 70000}]
)
def test_invalid_serving_config(kwargs):
    with pytest.raises(ConfigError):
        ServingConfig(**kwargs)


def test_default_serving_config():
    cfg = load_config()
    assert cfg.serving.model_source == "registry"
    assert cfg.serving.frozen_snapshot == "auto"  # follows approved promotions / rollbacks


@pytest.mark.parametrize(
    "kwargs", [{"background_size": 5}, {"top_k": 0}, {"top_k": 31}, {"additivity_tolerance": 0}]
)
def test_invalid_explainability_config(kwargs):
    with pytest.raises(ConfigError):
        ExplainabilityConfig(**kwargs)
