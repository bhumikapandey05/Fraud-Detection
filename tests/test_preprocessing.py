import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from fraud_detection.config import PreprocessingConfig
from fraud_detection.data.split import time_based_split
from fraud_detection.features.preprocessing import build_preprocessor, split_features_target
from fraud_detection.features.transformers import HourOfDayEncoder


@pytest.fixture
def splits(synthetic_df):
    return time_based_split(synthetic_df, 0.7, 0.15)


def test_output_features_and_no_target(splits):
    X, _ = split_features_target(splits.train)
    out = build_preprocessor().fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert "Class" not in out.columns
    assert "Time" not in out.columns  # raw counter dropped by default
    assert {"hour_sin", "hour_cos", "Amount", "V1", "V28"} <= set(out.columns)
    assert out.shape == (len(X), 31)


def test_statistics_are_learned_from_train_only(splits):
    """The fitted scaler must reflect TRAIN statistics, not the full dataset."""
    X_train, _ = split_features_target(splits.train)
    X_all, _ = split_features_target(pd.concat(splits.as_dict().values()))
    pre = build_preprocessor(PreprocessingConfig(scaler="standard")).fit(X_train)
    scaler = pre.named_transformers_["pca"]
    np.testing.assert_allclose(scaler.mean_, X_train[scaler.feature_names_in_].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X_all[scaler.feature_names_in_].mean().to_numpy())


def test_transform_is_deterministic_and_finite(splits):
    X_train, _ = split_features_target(splits.train)
    X_test, _ = split_features_target(splits.test)
    pre = build_preprocessor().fit(X_train)
    a, b = pre.transform(X_test), pre.transform(X_test)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.to_numpy()).all()


def test_transform_before_fit_raises(splits):
    X, _ = split_features_target(splits.test)
    with pytest.raises(NotFittedError):
        build_preprocessor().transform(X)


def test_config_variants(splits):
    X, _ = split_features_target(splits.train)
    cfg = PreprocessingConfig(amount_transform="none", add_hour_features=False, drop_raw_time=False)
    out = build_preprocessor(cfg).fit_transform(X)
    assert "Time" in out.columns and "hour_sin" not in out.columns


def test_log1p_is_applied_to_amount(splits):
    X, _ = split_features_target(splits.train)
    out = build_preprocessor().fit_transform(X)
    # After log1p + RobustScaler, the median amount maps to ~0.
    assert out["Amount"].median() == pytest.approx(0, abs=1e-9)


def test_preprocessor_is_cloneable():
    clone(build_preprocessor())  # required to work inside GridSearchCV / Pipelines


def test_split_features_target_missing_target(synthetic_df):
    with pytest.raises(KeyError):
        split_features_target(synthetic_df.drop(columns=["Class"]))


# --- HourOfDayEncoder --------------------------------------------------------
def test_hour_encoder_is_cyclical():
    enc = HourOfDayEncoder().fit(pd.DataFrame({"Time": [0.0]}))
    out = enc.transform(pd.DataFrame({"Time": [0.0, 86_400.0, 86_399.0, 43_200.0]}))
    np.testing.assert_allclose(out[0], out[1], atol=1e-12)  # same time on the next day
    assert np.linalg.norm(out[0] - out[2]) < 0.01  # 23:59:59 is next to 00:00
    np.testing.assert_allclose(out[3], [0, -1], atol=1e-12)  # noon is opposite midnight
    np.testing.assert_allclose((out**2).sum(axis=1), 1.0)


def test_hour_encoder_rejects_bad_input():
    enc = HourOfDayEncoder().fit(np.array([[0.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        enc.transform(np.array([[-5.0]]))
    with pytest.raises(ValueError):
        HourOfDayEncoder().fit(np.zeros((3, 2)))


def test_hour_encoder_feature_names():
    enc = HourOfDayEncoder().fit(pd.DataFrame({"Time": [1.0, 2.0]}))
    assert list(enc.get_feature_names_out()) == ["hour_sin", "hour_cos"]
