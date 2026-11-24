"""MLflow ``pyfunc`` wrapper so the registered model carries its threshold.

Registering only the scikit-learn pipeline would lose the tuned decision
threshold. This wrapper loads the saved :class:`FraudDetector` (pipeline +
threshold) and returns both the score and the decision, so anyone loading
``models:/fraud-detector@champion`` gets exactly the Phase 2 behaviour.
"""

from __future__ import annotations

import joblib
import mlflow.pyfunc
import pandas as pd


class FraudDetectorPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        self.detector = joblib.load(context.artifacts["detector"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        scores = self.detector.predict_proba(model_input)
        return pd.DataFrame(
            {
                "fraud_score": scores,
                "is_fraud": (scores >= self.detector.threshold).astype(int),
                "threshold": self.detector.threshold,
            }
        )
