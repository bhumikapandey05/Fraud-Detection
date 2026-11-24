import numpy as np
import pytest

from fraud_detection.config import PreprocessingConfig
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.models.detector import (
    DETECTOR_FILE,
    FraudDetector,
    load_detector,
    load_metadata,
    save_detector,
)
from fraud_detection.models.factory import build_training_pipeline


@pytest.fixture
def detector(synthetic_df):
    X, y = synthetic_df[FEATURE_COLUMNS], synthetic_df["Class"]
    pipe = build_training_pipeline(
        "logistic_regression", {"max_iter": 1000}, 0, PreprocessingConfig()
    ).fit(X, y)
    return FraudDetector(pipe, 0.3, "logistic_regression", FEATURE_COLUMNS)


def test_predict_uses_saved_threshold(detector, synthetic_df):
    scores = detector.predict_proba(synthetic_df)
    np.testing.assert_array_equal(detector.predict(synthetic_df), (scores >= 0.3).astype(int))
    detector.threshold = 0.9
    assert detector.predict(synthetic_df).sum() <= (scores >= 0.3).sum()


def test_extra_columns_ignored_and_column_order_irrelevant(detector, synthetic_df):
    reordered = synthetic_df[list(reversed(synthetic_df.columns))]  # includes the label
    np.testing.assert_array_equal(
        detector.predict_proba(reordered), detector.predict_proba(synthetic_df)
    )


def test_missing_columns_rejected(detector, synthetic_df):
    with pytest.raises(ValueError, match="Amount"):
        detector.predict(synthetic_df.drop(columns=["Amount"]))


def test_invalid_threshold_rejected(detector):
    with pytest.raises(ValueError):
        FraudDetector(detector.pipeline, 1.5, "x", FEATURE_COLUMNS)


def test_save_load_roundtrip(tmp_path, detector, synthetic_df):
    save_detector(detector, {"model_name": "logistic_regression"}, tmp_path)
    loaded = load_detector(tmp_path)
    assert loaded.threshold == detector.threshold
    np.testing.assert_array_equal(
        loaded.predict_proba(synthetic_df), detector.predict_proba(synthetic_df)
    )
    assert load_metadata(tmp_path)["model_name"] == "logistic_regression"


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_detector(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path)
    import joblib

    joblib.dump({"not": "a detector"}, tmp_path / DETECTOR_FILE)
    with pytest.raises(TypeError):
        load_detector(tmp_path)
