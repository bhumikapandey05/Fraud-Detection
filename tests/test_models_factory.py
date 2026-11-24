import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_detection.config import PreprocessingConfig, load_config
from fraud_detection.data.split import time_based_split
from fraud_detection.features.preprocessing import split_features_target
from fraud_detection.models.factory import (
    build_estimator,
    build_training_pipeline,
    compute_scale_pos_weight,
)

FAST_PARAMS = {
    "logistic_regression": {"max_iter": 2000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 20, "class_weight": "balanced_subsample", "n_jobs": -1},
    "xgboost": {"n_estimators": 30, "max_depth": 3, "scale_pos_weight": "auto", "n_jobs": -1},
}


@pytest.fixture
def data(synthetic_df):
    splits = time_based_split(synthetic_df, 0.7, 0.15)
    return split_features_target(splits.train), split_features_target(splits.validation)


def test_scale_pos_weight_is_negatives_over_positives():
    assert compute_scale_pos_weight([0, 0, 0, 1]) == 3.0
    with pytest.raises(ValueError):
        compute_scale_pos_weight([0, 0, 0])


def test_build_estimator_types_and_imbalance_settings():
    y = np.array([0] * 90 + [1] * 10)
    lr = build_estimator("logistic_regression", FAST_PARAMS["logistic_regression"], 1)
    rf = build_estimator("random_forest", FAST_PARAMS["random_forest"], 1)
    xgb = build_estimator("xgboost", FAST_PARAMS["xgboost"], 1, y_train=y)
    assert isinstance(lr, LogisticRegression) and lr.class_weight == "balanced"
    assert isinstance(rf, RandomForestClassifier) and rf.class_weight == "balanced_subsample"
    assert isinstance(xgb, XGBClassifier) and xgb.get_params()["scale_pos_weight"] == 9.0
    assert lr.random_state == rf.random_state == xgb.random_state == 1


def test_configured_params_are_valid_for_every_model():
    """The real config's hyperparameters must be accepted by each estimator."""
    cfg = load_config()
    for name in cfg.training.models:
        build_estimator(name, cfg.training.params.get(name), 0, y_train=[0, 0, 1])


def test_build_estimator_errors():
    with pytest.raises(ValueError, match="Unknown model"):
        build_estimator("svm", {}, 0)
    with pytest.raises(ValueError, match="y_train"):
        build_estimator("xgboost", {"scale_pos_weight": "auto"}, 0)


@pytest.mark.parametrize("name", list(FAST_PARAMS))
def test_pipeline_trains_on_raw_features_and_is_reproducible(name, data):
    (X_train, y_train), (X_val, _) = data

    def fit_and_score():
        pipe = build_training_pipeline(
            name, FAST_PARAMS[name], 42, PreprocessingConfig(), y_train=y_train
        )
        return pipe, pipe.fit(X_train, y_train).predict_proba(X_val)[:, 1]

    pipe, first = fit_and_score()
    _, second = fit_and_score()
    assert list(pipe.named_steps) == ["preprocessor", "model"]
    assert first.shape == (len(X_val),)
    assert ((first >= 0) & (first <= 1)).all()
    np.testing.assert_array_equal(first, second)  # same seed -> identical scores


def test_different_seed_changes_random_forest(data):
    (X_train, y_train), (X_val, _) = data
    scores = [
        build_training_pipeline(
            "random_forest", FAST_PARAMS["random_forest"], seed, PreprocessingConfig()
        )
        .fit(X_train, y_train)
        .predict_proba(X_val)[:, 1]
        for seed in (1, 2)
    ]
    assert not np.array_equal(*scores)
