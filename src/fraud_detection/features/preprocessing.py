"""Build the reusable preprocessing pipeline.

The returned object is an *unfitted* scikit-learn ``ColumnTransformer``. It
must be fitted on the training split only (``preprocessor.fit(X_train)``) and
then applied unchanged to validation and test data. In later phases it will be
chained with a model inside a single ``sklearn.pipeline.Pipeline`` so that the
exact same transformations are used during training and serving.

Column treatment:

* ``V1``-``V28``  -> scaled (already PCA outputs, but their spreads differ)
* ``Amount``      -> ``log1p`` (tames the heavy right tail) then scaled
* ``Time``        -> cyclical hour-of-day features; raw counter dropped
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from fraud_detection.config import PreprocessingConfig
from fraud_detection.data.schema import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    PCA_FEATURES,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_detection.features.transformers import HourOfDayEncoder


def _make_scaler(name: str):
    return RobustScaler() if name == "robust" else StandardScaler()


def build_preprocessor(cfg: PreprocessingConfig | None = None) -> ColumnTransformer:
    """Create the (unfitted) preprocessing transformer described in the config."""
    cfg = cfg or PreprocessingConfig()

    amount_steps = []
    if cfg.amount_transform == "log1p":
        amount_steps.append(
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        )
    amount_steps.append(("scale", _make_scaler(cfg.scaler)))

    transformers = [
        ("pca", _make_scaler(cfg.scaler), PCA_FEATURES),
        ("amount", Pipeline(amount_steps), [AMOUNT_COLUMN]),
    ]
    if cfg.add_hour_features:
        transformers.append(("hour", HourOfDayEncoder(), [TIME_COLUMN]))
    if not cfg.drop_raw_time:
        transformers.append(("time", _make_scaler(cfg.scaler), [TIME_COLUMN]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # anything not listed (e.g. the target) never reaches the model
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def split_features_target(
    df: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs ``X`` from the label ``y``."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found")
    return df[FEATURE_COLUMNS].copy(), df[target].astype("int64")
