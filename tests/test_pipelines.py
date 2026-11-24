"""End-to-end tests of the pipelines and CLI on a temporary synthetic project."""

import json

import pandas as pd
import pytest

from fraud_detection.cli import main
from fraud_detection.data.validation import DataValidationError
from fraud_detection.pipelines.prepare import load_preprocessor, run_prepare


def test_run_prepare_end_to_end(project_tmp):
    _, cfg = project_tmp
    report = run_prepare(cfg)

    for name in ("train", "validation", "test"):
        assert (cfg.paths.processed_dir / f"{name}.parquet").is_file()
    assert report["raw_validation"]["passed"]
    assert report["split_validation"]["passed"]
    assert report["preprocessor"]["fitted_on"] == "train"

    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    test = pd.read_parquet(cfg.paths.processed_dir / "test.parquet")
    assert train["Time"].max() < test["Time"].min()

    preprocessor = load_preprocessor(cfg.paths.artifacts_dir)
    out = preprocessor.transform(test.drop(columns=["Class"]))
    assert list(out.columns) == report["preprocessor"]["output_features"]
    saved = json.loads((cfg.paths.artifacts_dir / "data_preparation_report.json").read_text())
    assert saved["splits"]["train"]["rows"] == len(train)


def test_run_prepare_stops_on_invalid_data(project_tmp):
    _, cfg = project_tmp
    df = pd.read_csv(cfg.paths.raw_data)
    df.loc[0, "Class"] = 7
    df.to_csv(cfg.paths.raw_data, index=False)
    with pytest.raises(DataValidationError):
        run_prepare(cfg)
    assert not (cfg.paths.processed_dir / "train.parquet").exists()


def test_cli_commands(project_tmp, capsys):
    config_path, cfg = project_tmp
    assert main(["--config", str(config_path), "validate"]) == 0
    assert main(["--config", str(config_path), "eda"]) == 0
    assert (cfg.paths.eda_dir / "eda_report.md").is_file()
    assert main(["--config", str(config_path), "prepare"]) == 0
    assert main(["--config", str(config_path), "eda", "--on", "train"]) == 0
    assert "train" in capsys.readouterr().out


def test_cli_returns_error_when_dataset_missing(project_tmp):
    config_path, cfg = project_tmp
    cfg.paths.raw_data.unlink()
    assert main(["--config", str(config_path), "validate"]) == 1
