"""Build full training pipelines: Phase 1 preprocessing + a classifier.

Each model is wrapped together with a *fresh, unfitted* copy of the Phase 1
preprocessor in a single ``sklearn.pipeline.Pipeline``. Calling ``fit`` on the
training split therefore fits both the preprocessing and the model on training
data only, and the saved object can score raw transactions directly.

Class imbalance is handled with class weights derived from the training labels:

* Logistic Regression: ``class_weight="balanced"``
* Random Forest: ``class_weight="balanced_subsample"`` (re-weighted per tree)
* XGBoost: ``scale_pos_weight = negatives / positives`` in the training split

No rows are added, removed or duplicated, and validation/test data are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.config import SUPPORTED_MODELS, PreprocessingConfig
from fraud_detection.features.preprocessing import build_preprocessor

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def compute_scale_pos_weight(y_train) -> float:
    """Ratio of negatives to positives - XGBoost's standard imbalance weight."""
    y = np.asarray(y_train)
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("Training labels contain no positive (fraud) cases")
    return float((len(y) - positives) / positives)


def build_estimator(
    name: str, params: dict[str, Any] | None, random_state: int, y_train=None
) -> ClassifierMixin:
    """Create an unfitted classifier with the configured hyperparameters."""
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier  # imported lazily: optional native dependency

        if params.get("scale_pos_weight", "auto") == "auto":
            if y_train is None:
                raise ValueError("y_train is required to compute scale_pos_weight='auto'")
            params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
        return XGBClassifier(random_state=random_state, eval_metric="aucpr", **params)
    raise ValueError(f"Unknown model {name!r}; expected one of {SUPPORTED_MODELS}")


def build_training_pipeline(
    name: str,
    params: dict[str, Any] | None,
    random_state: int,
    preprocessing: PreprocessingConfig,
    y_train=None,
) -> Pipeline:
    """Return ``Pipeline([("preprocessor", ...), ("model", ...)])`` ready to ``fit``."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(preprocessing)),
            ("model", build_estimator(name, params, random_state, y_train)),
        ]
    )
