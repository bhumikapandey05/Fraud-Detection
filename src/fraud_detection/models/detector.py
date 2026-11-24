"""The deployable fraud detector: a fitted pipeline plus its decision threshold.

Saving the threshold *together with* the model avoids a classic production bug
where the service silently uses 0.5 instead of the tuned cut-off.

Artifacts written to ``model_dir``:

* ``fraud_detector.joblib`` - the :class:`FraudDetector` (preprocessing + model + threshold)
* ``model_metadata.json``   - human-readable details: model, threshold, metrics,
  data fingerprints and library versions
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

DETECTOR_FILE = "fraud_detector.joblib"
METADATA_FILE = "model_metadata.json"


class FraudDetector:
    """Score raw transactions and turn scores into fraud / not-fraud decisions."""

    def __init__(
        self, pipeline: Pipeline, threshold: float, model_name: str, feature_columns: list[str]
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.model_name = model_name
        self.feature_columns = list(feature_columns)

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        return X[self.feature_columns]  # ignores extra columns such as the label

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability (score) for each row, shape ``(n,)``."""
        return self.pipeline.predict_proba(self._select_features(X))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = flag as fraud, 0 = let through, using the tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def __repr__(self) -> str:
        return f"FraudDetector(model={self.model_name!r}, threshold={self.threshold:.4f})"


def save_detector(detector: FraudDetector, metadata: dict, model_dir: Path) -> dict[str, Path]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {"detector": model_dir / DETECTOR_FILE, "metadata": model_dir / METADATA_FILE}
    joblib.dump(detector, paths["detector"])
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    return paths


def load_detector(model_dir: Path) -> FraudDetector:
    path = Path(model_dir) / DETECTOR_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `fraud-detection train` first.")
    detector = joblib.load(path)
    if not isinstance(detector, FraudDetector):
        raise TypeError(f"{path} does not contain a FraudDetector")
    return detector


def load_metadata(model_dir: Path) -> dict:
    path = Path(model_dir) / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `fraud-detection train` first.")
    return json.loads(path.read_text())
