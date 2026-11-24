"""Custom scikit-learn transformers used by the preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

SECONDS_PER_DAY = 24 * 3600


class HourOfDayEncoder(TransformerMixin, BaseEstimator):
    """Turn ``Time`` (seconds since the first transaction) into hour-of-day features.

    Raw ``Time`` only ever increases, so values in the test period are larger
    than anything seen in training - a model would have to extrapolate. The
    time *of day*, however, repeats every 24 hours and can capture behaviour
    such as "fraud is relatively more common at night".

    The hour is encoded on a circle (sine + cosine) so that 23:00 and 00:00 are
    close together rather than 23 units apart.

    Note: the dataset does not state the clock time of the first transaction,
    so hour 0 is *relative* to the start of the recording.
    """

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True, dtype="float64")
        if X.shape[1] != 1:
            raise ValueError("HourOfDayEncoder expects exactly one column (Time)")
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = validate_data(self, X, reset=False, dtype="float64")
        if (X < 0).any():
            raise ValueError("Time values must be non-negative")
        angle = 2 * np.pi * (X % SECONDS_PER_DAY) / SECONDS_PER_DAY
        return np.hstack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        return np.asarray(["hour_sin", "hour_cos"], dtype=object)
