"""MLflow tracking + registry tests, using a throwaway local SQLite store per project."""

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_detection.cli import main
from fraud_detection.models.detector import load_detector, load_metadata
from fraud_detection.pipelines.freeze import frozen_path, verify_frozen
from fraud_detection.pipelines.track import registry_summary, track_frozen, track_training
from fraud_detection.tracking.mlflow_tracking import (
    TrackingError,
    configure_mlflow,
    data_version,
    flatten,
    load_registered_model,
)


def test_flatten_and_data_version():
    assert flatten({"a": {"b": 1, "c": {"d": "x"}}, "e": None}) == {
        "a.b": "1",
        "a.c.d": "x",
        "e": "None",
    }
    hashes = {"train": "aa", "validation": "bb", "test": "cc"}
    assert data_version(hashes) == data_version(dict(reversed(list(hashes.items()))))
    assert data_version(hashes) != data_version({**hashes, "test": "cd"})
    assert len(data_version(hashes)) == 12


def test_frozen_run_is_registered_as_champion(tracked_project):
    _, cfg, result = tracked_project
    assert result.registered_version == "1"
    summary = registry_summary(cfg)
    assert summary["aliases"] == {"champion": 1}
    version = summary["versions"][-1]
    meta = load_metadata(frozen_path(cfg, "v1") / "model")
    assert float(version["threshold"]) == pytest.approx(meta["threshold"]["value"], abs=1e-9)
    assert version["model_name"] == meta["model_name"]
    assert version["source"] == "frozen_snapshot"


def test_run_contains_params_metrics_data_and_artifacts(tracked_project):
    _, cfg, result = tracked_project
    configure_mlflow(cfg)
    client = MlflowClient()
    run = client.get_run(result.run_id)
    params, metrics = run.data.params, run.data.metrics
    meta = load_metadata(frozen_path(cfg, "v1") / "model")

    for key in (
        "split.strategy",
        "preprocessing.scaler",
        "preprocessing.amount_transform",
        "threshold.method",
        "threshold.cost_false_negative",
        "data_version",
        "random_state",
        "imbalance_strategy",
    ):
        assert key in params, key
    assert any(k.startswith("model.") for k in params)
    assert metrics["threshold"] == pytest.approx(meta["threshold"]["value"])
    assert metrics["val_pr_auc"] == pytest.approx(meta["validation_metrics"]["pr_auc"])
    assert metrics["test_pr_auc"] == pytest.approx(meta["test_metrics"]["pr_auc"])
    assert "test_pr_auc_ci_low" in metrics

    contexts = {i.tags[0].value for i in run.inputs.dataset_inputs}
    assert contexts == {"train", "validation", "test"}
    assert run.data.tags["data_version"] == params["data_version"]

    top = {a.path for a in client.list_artifacts(result.run_id)}
    assert {"reports", "model_metadata", "data", "config", "frozen"} <= top
    reports = {a.path for a in client.list_artifacts(result.run_id, "reports")}
    assert "reports/validation_vs_test.md" in reports

    children = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{result.run_id}'"
    )
    assert {c.data.tags["candidate"] for c in children} == set(meta["selection"]["candidates"])
    assert sum(c.data.tags["selected"] == "True" for c in children) == 1
    for child in children:
        assert "val_pr_auc" in child.data.metrics


def test_registered_champion_matches_frozen_detector(tracked_project):
    _, cfg, _ = tracked_project
    champion = load_registered_model(cfg)
    detector = load_detector(frozen_path(cfg, "v1") / "model")
    test = pd.read_parquet(cfg.paths.processed_dir / "test.parquet")
    out = champion.predict(test.drop(columns=["Class"]))
    assert list(out.columns) == ["fraud_score", "is_fraud", "threshold"]
    # Multi-threaded forests can differ in the last floating-point bit between calls,
    # so scores are compared with a tiny tolerance and decisions exactly.
    np.testing.assert_allclose(out["fraud_score"], detector.predict_proba(test), atol=1e-12)
    np.testing.assert_array_equal(out["is_fraud"], detector.predict(test))
    assert (out["threshold"] == detector.threshold).all()


def test_track_frozen_is_idempotent(tracked_project):
    _, cfg, first = tracked_project
    again = track_frozen(cfg, "v1")
    assert again.reused_existing_run
    assert again.run_id == first.run_id
    assert again.registered_version == first.registered_version
    assert len(registry_summary(cfg)["versions"]) >= 1
    assert verify_frozen(cfg, "v1")["ok"]  # tracking never modifies the snapshot


def test_new_training_runs_never_move_the_champion(tracked_project):
    config_path, cfg, frozen = tracked_project
    result = track_training(cfg, config_path, register=True)
    assert result.run_id != frozen.run_id
    summary = registry_summary(cfg)
    assert summary["aliases"]["champion"] == int(frozen.registered_version)
    assert int(result.registered_version) > int(frozen.registered_version)
    versions = {v["version"]: v for v in summary["versions"]}
    assert versions[int(result.registered_version)]["source"] == "train"


def test_refuses_to_log_when_data_changed(trained_tmp):
    config_path, cfg = trained_tmp
    path = cfg.paths.processed_dir / "validation.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    with pytest.raises(TrackingError, match="differ"):
        track_training(cfg, config_path)


def test_tracking_requires_mlflow_config(trained_tmp):
    config_path, cfg = trained_tmp
    cfg = cfg.__class__(**{**cfg.__dict__, "mlflow": None})
    with pytest.raises(TrackingError, match="mlflow"):
        track_training(cfg, config_path)


def test_cli_train_logs_to_mlflow_when_enabled(prepared_project):
    import yaml

    config_path, cfg = prepared_project
    raw = yaml.safe_load(config_path.read_text())
    raw["mlflow"]["log_training_runs"] = True
    raw["training"]["models"] = ["logistic_regression"]
    raw["training"]["bootstrap_samples"] = 0
    config_path.write_text(yaml.safe_dump(raw))

    assert main(["--config", str(config_path), "train"]) == 0
    assert main(["--config", str(config_path), "train", "--no-mlflow"]) == 0
    configure_mlflow(cfg)
    runs = MlflowClient().search_runs(
        [configure_mlflow(cfg)], "tags.source = 'train'", max_results=50
    )
    assert len(runs) == 1  # the --no-mlflow run was not logged
    assert registry_summary(cfg)["versions"] == []  # no registration without --register

    assert main(["--config", str(config_path), "freeze", "--name", "s"]) == 0
    assert main(["--config", str(config_path), "track-frozen", "--name", "s"]) == 0
    assert main(["--config", str(config_path), "registry"]) == 0
    assert registry_summary(cfg)["aliases"] == {"champion": 1}
