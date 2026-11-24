"""Request / response schemas: every transaction feature is validated.

Hard rules (violations -> HTTP 422, nothing is scored):

* all 30 features present (``Time``, ``V1``-``V28``, ``Amount``), no unknown fields
* real numbers only: no strings, booleans, ``null``, ``NaN`` or ``Infinity``
* ``Time`` >= 0 (seconds since the first transaction), ``Amount`` >= 0
* generous sanity limits that no genuine transaction should reach

Soft rule (still scored, but a warning is returned): a value outside the range
seen in the *training* split, where the model has never been tested.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_detection.data.schema import AMOUNT_COLUMN, FEATURE_COLUMNS, PCA_FEATURES, TIME_COLUMN

MAX_TIME_SECONDS = 10 * 365 * 24 * 3600  # ten years after the reference point
MAX_AMOUNT = 1_000_000.0
MAX_ABS_PCA = 1_000.0  # training values lie within about +/-120

STRICT = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _StrictModel(BaseModel):
    model_config = STRICT


_FEATURE_FIELDS: dict = {
    TIME_COLUMN: (
        float,
        Field(ge=0, le=MAX_TIME_SECONDS, description="Seconds since the first transaction"),
    ),
    **{
        name: (float, Field(ge=-MAX_ABS_PCA, le=MAX_ABS_PCA, description="PCA component"))
        for name in PCA_FEATURES
    },
    AMOUNT_COLUMN: (float, Field(ge=0, le=MAX_AMOUNT, description="Transaction amount")),
}

Transaction = create_model(
    "Transaction",
    __base__=_StrictModel,
    transaction_id=(
        str | None,
        Field(
            None,
            max_length=64,
            pattern=r"^[A-Za-z0-9_.:-]+$",
            description="Optional client reference, echoed back",
        ),
    ),
    **_FEATURE_FIELDS,
)


class BatchRequest(_StrictModel):
    transactions: list[Transaction] = Field(min_length=1)  # type: ignore[valid-type]


def to_frame(transactions: list) -> pd.DataFrame:
    """Validated request objects -> feature DataFrame in training column order."""
    return pd.DataFrame([t.model_dump(include=set(FEATURE_COLUMNS)) for t in transactions])[
        FEATURE_COLUMNS
    ]


class ModelRef(BaseModel):
    name: str
    version: str | None
    source: str
    threshold: float


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_score: float = Field(description="Model probability of fraud, 0-1")
    is_fraud: bool
    decision: Literal["REVIEW", "APPROVE"]
    warnings: list[str] = []


class PredictionResponse(Prediction):
    model: ModelRef


class BatchResponse(BaseModel):
    model: ModelRef
    n_transactions: int
    n_flagged: int
    predictions: list[Prediction]


class FeatureRanges:
    """Per-feature min/max of the TRAINING split, for out-of-range warnings."""

    def __init__(self, low: dict[str, float], high: dict[str, float]) -> None:
        self.low, self.high = low, high

    @classmethod
    def from_training_split(cls, path) -> FeatureRanges | None:
        try:
            train = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        except (FileNotFoundError, OSError):
            return None
        return cls(train.min().to_dict(), train.max().to_dict())

    def warnings_for(self, row: dict) -> list[str]:
        out = []
        for col in FEATURE_COLUMNS:
            if col == TIME_COLUMN:  # Time is used only as hour-of-day; later times are normal
                continue
            value = row[col]
            if value < self.low[col] or value > self.high[col]:
                out.append(
                    f"{col}={value:g} is outside the training range "
                    f"[{self.low[col]:.4g}, {self.high[col]:.4g}]; the score is less reliable"
                )
        return out


class FeatureContribution(BaseModel):
    feature: str
    label: str = Field(description="Display name; V1-V28 are labelled as anonymized")
    value: float = Field(description="The submitted value")
    display_value: str
    contribution: float = Field(description="SHAP contribution to the fraud score")
    direction: Literal["toward_fraud", "away_from_fraud", "neutral"]


class ExplanationMethod(BaseModel):
    method: str
    output: str
    baseline_meaning: str
    background_size: int
    background_random_state: int
    notes: list[str]


class ExplanationResponse(Prediction):
    model: ModelRef
    base_value: float = Field(description="Average fraud score over the background sample")
    top_toward_fraud: list[FeatureContribution]
    top_away_from_fraud: list[FeatureContribution]
    other_features_contribution: float = Field(
        description="Sum of contributions of the features not listed in the two top lists"
    )
    contributions: list[FeatureContribution] = Field(
        description="All 30 transaction fields, sorted by absolute contribution"
    )
    additivity_error: float = Field(
        description="|base_value + sum(contributions) - fraud_score|; ~1e-7 or less"
    )
    explanation_method: ExplanationMethod
