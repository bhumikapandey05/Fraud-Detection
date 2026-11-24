import json
import os
import stat

import pandas as pd
import pytest

from fraud_detection.cli import main
from fraud_detection.pipelines.evaluate import run_evaluate
from fraud_detection.pipelines.freeze import (
    MANIFEST_FILE,
    FrozenModelError,
    freeze_model,
    frozen_path,
    load_manifest,
    verify_frozen,
)


def test_freeze_creates_complete_read_only_snapshot(trained_tmp):
    config_path, cfg = trained_tmp
    root = freeze_model(cfg, "snap", config_path)
    manifest = load_manifest(cfg, "snap")
    for rel in (
        "model/fraud_detector.joblib",
        "model/model_metadata.json",
        "reports/evaluation_report.md",
        "reports/validation_vs_test.md",
        "reports/validation_vs_test.json",
        "data/data_preparation_report.json",
        "config/config.yaml",
    ):
        assert rel in manifest["files"], rel
        mode = (root / rel).stat().st_mode
        assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), f"{rel} is writable"
    assert (
        manifest["threshold"]
        == json.loads((root / "model/model_metadata.json").read_text())["threshold"]["value"]
    )
    assert not list(cfg.paths.frozen_dir.glob(".*.tmp"))  # no half-written leftovers


def test_snapshot_survives_retraining_and_cleaning(trained_tmp):
    """The frozen copy is independent of the working model folder."""
    config_path, cfg = trained_tmp
    freeze_model(cfg, "snap", config_path)
    import shutil

    shutil.rmtree(cfg.paths.model_dir)  # e.g. `make clean` or a new training run
    assert verify_frozen(cfg, "snap")["ok"]
    assert run_evaluate(cfg, model_dir=frozen_path(cfg, "snap") / "model")["reproducible"]


def test_freeze_never_overwrites(trained_tmp):
    config_path, cfg = trained_tmp
    freeze_model(cfg, "snap", config_path)
    with pytest.raises(FileExistsError, match="never overwritten"):
        freeze_model(cfg, "snap", config_path)


@pytest.mark.parametrize("name", ["", "../escape", ".hidden"])
def test_invalid_snapshot_names(trained_tmp, name):
    config_path, cfg = trained_tmp
    with pytest.raises(ValueError):
        freeze_model(cfg, name, config_path)


def test_verify_detects_modified_file(trained_tmp):
    config_path, cfg = trained_tmp
    root = freeze_model(cfg, "snap", config_path)
    target = root / "model" / "model_metadata.json"
    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    meta = json.loads(target.read_text())
    meta["threshold"]["value"] = 0.9
    target.write_text(json.dumps(meta))
    with pytest.raises(FrozenModelError, match="modified"):
        verify_frozen(cfg, "snap")


def test_verify_detects_changed_split_data(trained_tmp):
    config_path, cfg = trained_tmp
    freeze_model(cfg, "snap", config_path)
    path = cfg.paths.processed_dir / "test.parquet"
    pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
    with pytest.raises(FrozenModelError, match="split files differ"):
        verify_frozen(cfg, "snap")


def test_verify_missing_snapshot(project_tmp):
    _, cfg = project_tmp
    with pytest.raises(FrozenModelError, match="not found"):
        verify_frozen(cfg, "nope")


def test_freeze_refuses_model_that_does_not_reproduce(trained_tmp):
    config_path, cfg = trained_tmp
    path = cfg.paths.model_dir / "model_metadata.json"
    meta = json.loads(path.read_text())
    meta["test_metrics"]["recall"] = 0.0
    path.write_text(json.dumps(meta))
    with pytest.raises(FrozenModelError, match="not reproduced"):
        freeze_model(cfg, "snap", config_path)
    assert not frozen_path(cfg, "snap").exists()


def test_cli_freeze_verify_gap_report(trained_tmp):
    config_path, cfg = trained_tmp
    args = ["--config", str(config_path)]
    assert main([*args, "gap-report"]) == 0
    assert (cfg.paths.training_report_dir / "validation_vs_test.md").is_file()
    assert main([*args, "freeze", "--name", "cli-snap"]) == 0
    assert main([*args, "freeze", "--name", "cli-snap"]) == 1  # refuses to overwrite
    assert main([*args, "verify-frozen", "--name", "cli-snap"]) == 0
    model_dir = frozen_path(cfg, "cli-snap") / "model"
    assert main([*args, "evaluate", "--model-dir", str(model_dir)]) == 0
    assert (frozen_path(cfg, "cli-snap") / MANIFEST_FILE).is_file()
