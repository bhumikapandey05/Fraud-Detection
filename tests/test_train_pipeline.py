"""End-to-end tests of `train` / `evaluate` on a synthetic project (trained once per session)."""

import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.cli import main
from fraud_detection.models.detector import load_detector, load_metadata
from fraud_detection.pipelines import train as train_module
from fraud_detection.pipelines.evaluate import run_evaluate
from fraud_detection.pipelines.train import _file_sha256


def test_artifacts_and_reports_exist(trained_project):
    _, cfg, _ = trained_project
    assert (cfg.paths.model_dir / "fraud_detector.joblib").is_file()
    assert (cfg.paths.model_dir / "model_metadata.json").is_file()
    report_dir = cfg.paths.training_report_dir
    for name in (
        "evaluation_report.md",
        "model_comparison.csv",
        "threshold_tradeoff_validation.csv",
    ):
        assert (report_dir / name).is_file()
    assert len(list((report_dir / "figures").glob("*.png"))) == 4
    comparison = pd.read_csv(report_dir / "model_comparison.csv")
    assert len(comparison) == 3 and comparison["selected"].sum() == 1


def test_metadata_records_selection_threshold_and_data(trained_project):
    _, cfg, hashes = trained_project
    meta = load_metadata(cfg.paths.model_dir)
    candidates = meta["selection"]["candidates"]
    assert set(candidates) == {"logistic_regression", "random_forest", "xgboost"}
    # Selected = best validation PR-AUC.
    assert meta["model_name"] == max(candidates, key=lambda m: candidates[m]["pr_auc"])
    assert meta["threshold"]["tuned_on"] == "validation"
    assert meta["threshold"]["value"] == candidates[meta["model_name"]]["threshold"]
    assert meta["data"]["files_sha256"] == hashes
    assert meta["data"]["preprocessor_matches_phase1_artifact"] is True
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        assert key in meta["test_metrics"]
    assert {"python", "scikit-learn", "xgboost"} <= set(meta["environment"])


def test_training_never_modifies_the_splits(trained_project):
    _, cfg, hashes_before = trained_project
    for name, digest in hashes_before.items():
        assert _file_sha256(cfg.paths.processed_dir / f"{name}.parquet") == digest


def test_saved_detector_uses_threshold(trained_project):
    _, cfg, _ = trained_project
    detector = load_detector(cfg.paths.model_dir)
    test = pd.read_parquet(cfg.paths.processed_dir / "test.parquet")
    preds = detector.predict(test)
    np.testing.assert_array_equal(
        preds, (detector.predict_proba(test) >= detector.threshold).astype(int)
    )


def test_evaluate_reproduces_recorded_test_metrics(trained_project):
    config_path, cfg, _ = trained_project
    result = run_evaluate(cfg)
    assert result["reproducible"], result["mismatches"]
    assert main(["--config", str(config_path), "evaluate"]) == 0
    assert main(["--config", str(config_path), "verify"]) == 0


def test_evaluate_flags_tampered_metadata(trained_project, tmp_path):
    _, cfg, _ = trained_project
    path = cfg.paths.model_dir / "model_metadata.json"
    original = path.read_text()
    try:
        meta = json.loads(original)
        meta["test_metrics"]["pr_auc"] += 0.1
        path.write_text(json.dumps(meta))
        result = run_evaluate(cfg)
        assert not result["reproducible"]
        assert "pr_auc" in result["mismatches"]
    finally:
        path.write_text(original)


def test_test_split_is_scored_once_and_only_for_the_selected_model(prepared_project, monkeypatch):
    """Spy on metric calls: exactly one call may use the test labels."""
    _, cfg = prepared_project
    y_test = pd.read_parquet(cfg.paths.processed_dir / "test.parquet")["Class"].to_numpy()
    calls = []
    real = train_module.classification_metrics

    def spy(y_true, scores, threshold, **kwargs):
        calls.append(np.array_equal(np.asarray(y_true), y_test))
        return real(y_true, scores, threshold, **kwargs)

    monkeypatch.setattr(train_module, "classification_metrics", spy)
    fast = cfg.training.__class__(
        random_state=cfg.training.random_state,
        models=cfg.training.models,
        bootstrap_samples=0,
        params=cfg.training.params,
    )
    result = train_module.run_train(cfg.__class__(**{**cfg.__dict__, "training": fast}))
    assert sum(calls) == 1  # the test set was evaluated exactly once
    assert len(calls) == 2 * 3 + 1  # 2 validation evaluations per model + 1 test
    assert result["selected_model"] in cfg.training.models


def test_train_fails_cleanly_without_prepared_data(project_tmp):
    config_path, _ = project_tmp
    assert main(["--config", str(config_path), "train"]) == 1


@pytest.mark.parametrize("method", ["cost", "f_beta"])
def test_both_threshold_methods_run(prepared_project, method):
    _, cfg = prepared_project
    from fraud_detection.config import ThresholdConfig, TrainingConfig

    cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "threshold": ThresholdConfig(method=method),
            "training": TrainingConfig(
                models=("logistic_regression",),
                bootstrap_samples=0,
                params=cfg.training.params,
            ),
        }
    )
    result = train_module.run_train(cfg)
    assert 0 <= result["threshold"] <= 1
