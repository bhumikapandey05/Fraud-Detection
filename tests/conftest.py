"""Shared pytest fixtures.

Unit tests use a SMALL SYNTHETIC table that has the same schema as the Kaggle
file. It exists only so tests are fast and run anywhere (e.g. CI) without the
150 MB dataset. It is never used by the real pipelines. Tests that need the
real data are marked ``integration`` and skipped when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraud_detection.config import load_config
from fraud_detection.data.schema import EXPECTED_COLUMNS, PCA_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def make_synthetic_transactions(n_rows: int = 4000, fraud_rate: float = 0.05, seed: int = 0):
    """Create a schema-compatible fake dataset for unit tests only."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_rows) < fraud_rate).astype("int64")
    data = {"Time": np.sort(rng.uniform(0, 172_000, n_rows)).round()}
    for i, col in enumerate(PCA_FEATURES):
        shift = -2.0 * y if i < 5 else 0.0  # make a few features informative
        data[col] = rng.normal(0, 1 + i / 28, n_rows) + shift
    data["Amount"] = rng.lognormal(3, 1.2, n_rows).round(2)
    data["Class"] = y
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_transactions()


def make_project(root: Path, df: pd.DataFrame) -> Path:
    """Write a synthetic raw CSV and a fast config under ``root``; return the config path."""
    raw = root / "data" / "raw" / "creditcard.csv"
    raw.parent.mkdir(parents=True)
    df.to_csv(raw, index=False)

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text())
    config["dataset"]["expected_rows"] = None
    config["split"]["min_positives_per_split"] = 10
    # Small models + few bootstrap resamples keep the test suite fast.
    config["training"]["bootstrap_samples"] = 50
    config["training"]["params"]["random_forest"]["n_estimators"] = 30
    config["training"]["params"]["xgboost"]["n_estimators"] = 50
    # Absolute paths so the CLI (which resolves against the real project root) uses `root`.
    config["paths"] = {key: str(root / value) for key, value in config["paths"].items()}
    for key in ("tracking_db", "artifact_dir"):
        config["mlflow"][key] = str(root / config["mlflow"][key])
    config["mlflow"]["log_training_runs"] = False  # MLflow tests switch it on explicitly
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture
def project_tmp(tmp_path: Path, synthetic_df: pd.DataFrame):
    """A temporary project with a synthetic raw CSV. Returns (config_path, Config)."""
    config_path = make_project(tmp_path, synthetic_df)
    return config_path, load_config(config_path)


@pytest.fixture
def prepared_project(project_tmp):
    """A temporary project on which `prepare` has already run."""
    from fraud_detection.pipelines.prepare import run_prepare

    run_prepare(project_tmp[1])
    return project_tmp


@pytest.fixture(scope="session")
def trained_project(tmp_path_factory):
    """Prepare + train ONCE per test session (via the CLI) and share the result.

    Returns (config_path, Config, split_hashes_before_training).
    """
    from fraud_detection.cli import main
    from fraud_detection.pipelines.train import _file_sha256

    root = tmp_path_factory.mktemp("trained")
    config_path = make_project(root, make_synthetic_transactions())
    cfg = load_config(config_path)
    assert main(["--config", str(config_path), "prepare"]) == 0
    hashes = {
        name: _file_sha256(cfg.paths.processed_dir / f"{name}.parquet")
        for name in ("train", "validation", "test")
    }
    assert main(["--config", str(config_path), "train"]) == 0
    return config_path, cfg, hashes


@pytest.fixture
def trained_tmp(prepared_project):
    """Function-scoped prepared + trained project (fast) for tests that modify files."""
    from fraud_detection.config import TrainingConfig
    from fraud_detection.pipelines.train import run_train

    config_path, cfg = prepared_project
    fast = TrainingConfig(
        models=("logistic_regression", "random_forest"),
        bootstrap_samples=0,
        params=cfg.training.params,
    )
    cfg = cfg.__class__(**{**cfg.__dict__, "training": fast})
    run_train(cfg)
    return config_path, cfg


@pytest.fixture(scope="session")
def tracked_project(trained_project):
    """Session project whose model is frozen as 'v1' and tracked + registered in MLflow."""
    from fraud_detection.pipelines.freeze import freeze_model
    from fraud_detection.pipelines.track import track_frozen

    config_path, cfg, _ = trained_project
    freeze_model(cfg, "v1", config_path)
    result = track_frozen(cfg, "v1")
    return config_path, cfg, result


@pytest.fixture(scope="session")
def serving_cfg(tracked_project):
    """Config whose serving section points at the session's frozen + registered 'v1'."""
    from dataclasses import replace

    config_path, cfg, _ = tracked_project
    return config_path, replace(
        cfg, serving=replace(cfg.serving, frozen_snapshot="v1", max_batch_size=50)
    )


@pytest.fixture(scope="session")
def api_client(serving_cfg):
    """A running TestClient for the API (lifespan executed, model loaded)."""
    from fastapi.testclient import TestClient

    from fraud_detection.serving.api import create_app

    with TestClient(create_app(serving_cfg[1])) as client:
        yield client


def make_sandbox(tmp_path_factory, serving_cfg, name: str, lenient: bool = False):
    """Isolated copy of the session's frozen 'v1' with its own registry + monitoring store."""
    import yaml

    from fraud_detection.pipelines.demo import setup_sandbox

    config_path, cfg = serving_cfg
    workdir = tmp_path_factory.mktemp(name)
    sandbox_path, sandbox_cfg = setup_sandbox(
        workdir / "sb", base=cfg, snapshot="v1", base_config_path=config_path
    )
    raw = yaml.safe_load(sandbox_path.read_text())
    raw["monitoring"]["min_rows"] = 50
    if lenient:  # make any reasonable candidate eligible, to exercise promotion
        raw["retraining"]["criteria"] = {
            "max_validation_pr_auc_drop": 1.0,
            "max_validation_recall_drop": 1.0,
            "max_validation_cost_increase": 100.0,
            "min_holdout_pr_auc_gain": -1.0,
            "require_holdout": False,
        }
    sandbox_path.write_text(yaml.safe_dump(raw))
    return sandbox_path, load_config(sandbox_path)
