import pandas as pd
import pytest

from fraud_detection.data.validation import DataValidationError
from fraud_detection.data.verification import (
    check_preprocessor_consistency,
    load_splits,
    verify_prepared_data,
)
from fraud_detection.features.preprocessing import build_preprocessor, split_features_target


def test_verification_passes_after_prepare(prepared_project):
    _, cfg = prepared_project
    verified = verify_prepared_data(cfg)
    assert verified.report.passed
    assert set(verified.summary) == {"train", "validation", "test"}


def test_detects_counts_that_differ_from_phase1(prepared_project):
    _, cfg = prepared_project
    path = cfg.paths.processed_dir / "test.parquet"
    test = pd.read_parquet(path)
    test.drop(index=test.index[test["Class"] == 1][:1]).to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match="Phase 1 recorded"):
        verify_prepared_data(cfg)


def test_detects_rows_leaking_between_splits(prepared_project):
    _, cfg = prepared_project
    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    path = cfg.paths.processed_dir / "validation.parquet"
    leaked = pd.concat([pd.read_parquet(path), train.tail(3)], ignore_index=True)
    leaked.to_parquet(path, index=False)
    with pytest.raises(DataValidationError, match="identical rows|Time leakage"):
        verify_prepared_data(cfg)


def test_missing_splits_raise(project_tmp):
    _, cfg = project_tmp
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_splits(cfg.paths.processed_dir)


def test_preprocessor_consistency(prepared_project, tmp_path):
    _, cfg = prepared_project
    splits = load_splits(cfg.paths.processed_dir)
    X_train, _ = split_features_target(splits.train)
    X_val, _ = split_features_target(splits.validation)
    same = build_preprocessor(cfg.preprocessing).fit(X_train)
    assert check_preprocessor_consistency(same, cfg.paths.artifacts_dir, X_val) is True
    different = build_preprocessor(cfg.preprocessing).fit(X_val)  # fitted on the wrong data
    assert check_preprocessor_consistency(different, cfg.paths.artifacts_dir, X_val) is False
    assert check_preprocessor_consistency(same, tmp_path, X_val) is None
