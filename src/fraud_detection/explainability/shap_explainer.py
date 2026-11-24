"""Local SHAP explanations for the frozen Random Forest.

What is explained
-----------------
The number the API shows: the model's **probability of fraud** (class 1) from
``FraudDetector.predict_proba``. A Random Forest's probability is the average of
its trees' leaf probabilities, so TreeSHAP on the forest's raw output *is* the
probability space - there is no log-odds link to undo (unlike e.g. XGBoost).
Every explanation satisfies, and is checked against::

    baseline + sum(contributions) == displayed fraud score   (to ~1e-7)

Baseline
--------
SHAP's default "tree_path_dependent" mode uses the training weights stored in the
trees. Because the model was trained with balanced class weights, that baseline
is 0.5 - "a 50/50 world" - which would badly mislead a reader. We therefore use
**interventional** TreeSHAP with a fixed, seeded background sample of real
*training* transactions (natural fraud rate, ~0.2%): the baseline is then exactly
the model's average score over those transactions, i.e. "a typical transaction".
The background split must be byte-identical to the one the frozen model was
trained on (checked against the snapshot's SHA-256), and validation/test data are
never used.

From model inputs back to transaction fields
--------------------------------------------
The trees see 31 preprocessed features; people see 30 transaction fields. SHAP
values are additive, so each field's contribution is the **sum** of the
contributions of the features derived from it:

* ``V1``-``V28`` -> their scaled copies (one-to-one)
* ``Amount``     -> ``log1p``-scaled Amount (one-to-one)
* ``Time``       -> ``hour_sin`` + ``hour_cos`` (the model only sees hour of day)

Contributions describe how the *model* used each value for this transaction.
They are not causal effects, and ``V1``-``V28`` are anonymised PCA components
whose business meaning is not disclosed - they are labelled as such.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from fraud_detection.data.schema import AMOUNT_COLUMN, FEATURE_COLUMNS, PCA_FEATURES, TIME_COLUMN
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.detector import FraudDetector

logger = get_logger(__name__)

SECONDS_PER_HOUR = 3600


class ExplanationError(RuntimeError):
    """Explanations cannot be produced (or would not be consistent with the score)."""


# ----------------------------------------------------------------------------------------- #
# Labels: honest names, no invented meanings
# ----------------------------------------------------------------------------------------- #
def feature_label(name: str) -> str:
    if name in PCA_FEATURES:
        return f"{name} (anonymized PCA component)"
    if name == AMOUNT_COLUMN:
        return "Amount (transaction amount)"
    if name == TIME_COLUMN:
        return "Time (used as hour of day)"
    return name


def feature_description(name: str) -> str:
    if name in PCA_FEATURES:
        return (
            "Anonymized principal component published by the data provider; its business "
            "meaning is not disclosed, so it is not interpreted here."
        )
    if name == AMOUNT_COLUMN:
        return "Transaction amount (the model uses its logarithm, scaled)."
    if name == TIME_COLUMN:
        return (
            "Seconds since the first transaction in the dataset. The model only uses the "
            "hour of day derived from it (relative to the start of the recording)."
        )
    return ""


def display_value(name: str, value: float) -> str:
    if name == TIME_COLUMN:
        hour = int(value // SECONDS_PER_HOUR) % 24
        return f"{value:,.0f} s (hour {hour:02d} of the relative day)"
    if name == AMOUNT_COLUMN:
        return f"{value:,.2f}"
    return f"{value:.4f}"


# ----------------------------------------------------------------------------------------- #
# Preprocessed feature -> original transaction field
# ----------------------------------------------------------------------------------------- #
def output_to_input_map(preprocessor: ColumnTransformer) -> dict[str, str]:
    """Map every preprocessor output column to the single raw field it comes from.

    Works for any transformer in the ColumnTransformer that either keeps columns
    one-to-one or derives several outputs from one input column (e.g. Time ->
    hour_sin, hour_cos). Anything else is rejected rather than guessed.
    """
    mapping: dict[str, str] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        columns = list(columns)
        outputs = list(transformer.get_feature_names_out(columns))
        if len(columns) == len(outputs) and all(o in columns for o in outputs):
            mapping.update({o: o for o in outputs})  # one-to-one, same names
        elif len(columns) == 1:
            mapping.update({o: columns[0] for o in outputs})  # several outputs from one input
        else:
            raise ExplanationError(f"Cannot attribute outputs of transformer '{name}' to inputs")
    return mapping


# ----------------------------------------------------------------------------------------- #
# Explanation objects
# ----------------------------------------------------------------------------------------- #
@dataclass
class Contribution:
    feature: str
    label: str
    value: float
    display_value: str
    contribution: float

    @property
    def direction(self) -> str:
        if self.contribution > 0:
            return "toward_fraud"
        return "away_from_fraud" if self.contribution < 0 else "neutral"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": self.value,
            "display_value": self.display_value,
            "contribution": self.contribution,
            "direction": self.direction,
        }


@dataclass
class Explanation:
    fraud_score: float
    base_value: float
    contributions: list[Contribution]  # all 30 fields, sorted by |contribution|, descending
    additivity_error: float
    model_features: dict[str, float] = field(default_factory=dict)  # 31 preprocessed values

    def toward_fraud(self, k: int) -> list[Contribution]:
        return [c for c in self.contributions if c.contribution > 0][:k]

    def away_from_fraud(self, k: int) -> list[Contribution]:
        return [c for c in self.contributions if c.contribution < 0][:k]

    def contribution_of(self, feature: str) -> float:
        return next(c.contribution for c in self.contributions if c.feature == feature)


# ----------------------------------------------------------------------------------------- #
# Explainer
# ----------------------------------------------------------------------------------------- #
class FraudExplainer:
    """Interventional TreeSHAP for a ``FraudDetector`` (preprocessing + forest)."""

    METHOD = "TreeSHAP (interventional) on the Random Forest's fraud probability"

    def __init__(
        self,
        detector: FraudDetector,
        background: pd.DataFrame,
        additivity_tolerance: float = 1e-6,
        background_info: dict | None = None,
    ) -> None:
        import shap  # imported here: optional for everything except explanations

        pipeline: Pipeline = detector.pipeline
        self.detector = detector
        self.preprocessor = pipeline[:-1]
        self.model = pipeline[-1]
        if not hasattr(self.model, "estimators_"):
            raise ExplanationError("Explanations are implemented for tree ensembles only")
        classes = list(self.model.classes_)
        if 1 not in classes:
            raise ExplanationError("Model has no fraud class (1)")
        self.class_index = classes.index(1)

        last = self.preprocessor[-1] if isinstance(self.preprocessor, Pipeline) else None
        self.output_map = output_to_input_map(last if last is not None else self.preprocessor)
        self.model_feature_names = list(self.preprocessor.get_feature_names_out())
        missing = set(FEATURE_COLUMNS) - set(self.output_map.values())
        if missing:
            raise ExplanationError(f"Fields with no model feature: {sorted(missing)}")

        self.tolerance = additivity_tolerance
        bg = self.preprocessor.transform(background[FEATURE_COLUMNS])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainer = shap.TreeExplainer(
                self.model,
                data=shap.maskers.Independent(bg, max_samples=len(bg)),  # use ALL rows given
                feature_perturbation="interventional",
                model_output="raw",  # = predict_proba for a scikit-learn Random Forest
            )
        self.base_value = float(np.atleast_1d(self._explainer.expected_value)[self.class_index])
        self.background_info = {
            "size": int(len(background)),
            "mean_model_score": float(detector.predict_proba(background).mean()),
            **(background_info or {}),
        }
        self._lock = threading.Lock()

        # Self-check: the baseline must be the model's average score on the background.
        if abs(self.base_value - self.background_info["mean_model_score"]) > self.tolerance:
            raise ExplanationError("SHAP baseline differs from the model's mean background score")

    def explain(self, features: pd.DataFrame) -> list[Explanation]:
        """Explain each row; raises if SHAP values do not add up to the model's score."""
        features = features[FEATURE_COLUMNS]
        transformed = self.preprocessor.transform(features)
        with self._lock, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = self._explainer.shap_values(transformed, check_additivity=False)
        values = np.asarray(values)
        phi = values[..., self.class_index] if values.ndim == 3 else values  # (n, n_model_feat)

        # The score shown to users comes from the same call as /predict.
        scores = self.detector.predict_proba(features)
        errors = np.abs(self.base_value + phi.sum(axis=1) - scores)
        if (errors > self.tolerance).any():
            raise ExplanationError(
                f"SHAP values do not add up to the model score (max error {errors.max():.2e})"
            )

        per_field = (
            pd.DataFrame(phi, columns=self.model_feature_names)
            .T.groupby(lambda col: self.output_map[col])
            .sum()
            .T[FEATURE_COLUMNS]
        )

        explanations = []
        for i in range(len(features)):
            row = features.iloc[i]
            contribs = [
                Contribution(
                    feature=name,
                    label=feature_label(name),
                    value=float(row[name]),
                    display_value=display_value(name, float(row[name])),
                    contribution=float(per_field.iloc[i][name]),
                )
                for name in FEATURE_COLUMNS
            ]
            contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
            explanations.append(
                Explanation(
                    fraud_score=float(scores[i]),
                    base_value=self.base_value,
                    contributions=contribs,
                    additivity_error=float(errors[i]),
                    model_features=dict(
                        zip(self.model_feature_names, map(float, transformed.iloc[i]), strict=True)
                    ),
                )
            )
        return explanations


def sample_background(train: pd.DataFrame, size: int, random_state: int) -> pd.DataFrame:
    """Seeded random sample of training transactions (natural class mix)."""
    if len(train) < size:
        raise ExplanationError(f"Training split has only {len(train)} rows (< {size})")
    return train.sample(size, random_state=random_state)[FEATURE_COLUMNS].reset_index(drop=True)
