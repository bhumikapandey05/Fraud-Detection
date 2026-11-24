"""Reference data for monitoring, cached per model version.

* Feature reference: a seeded random sample of the TRAINING split - "what the model
  learned from".
* Score reference: the model's scores on the VALIDATION split. Scores on training
  rows would be over-confident (the forest has seen them), so they are a poor
  baseline for production scores; validation rows are out-of-sample.
* Expected review rate: share of validation scores at or above the model's threshold.

The test split is never read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.models.detector import FraudDetector
from fraud_detection.pipelines.train import _file_sha256


@dataclass
class Reference:
    features: pd.DataFrame
    validation_scores: pd.Series
    expected_alert_rate: float
    train_low: dict[str, float]
    train_high: dict[str, float]
    info: dict


def build_reference(
    cfg: Config, detector: FraudDetector, model_version: str, detector_sha256: str
) -> Reference:
    mcfg = cfg.monitoring
    train_path = cfg.paths.processed_dir / "train.parquet"
    val_path = cfg.paths.processed_dir / "validation.parquet"
    key = {
        "model_version": model_version,
        "detector_sha256": detector_sha256,
        "train_sha256": _file_sha256(train_path),
        "validation_sha256": _file_sha256(val_path),
        "reference_size": mcfg.reference_size,
        "random_state": mcfg.random_state,
    }
    cache = Path(cfg.paths.monitoring_dir) / "reference" / f"v{model_version}"
    info_path = cache / "info.json"
    if info_path.is_file() and json.loads(info_path.read_text())["key"] == key:
        info = json.loads(info_path.read_text())
        features = pd.read_parquet(cache / "features.parquet")
        scores = pd.read_parquet(cache / "validation_scores.parquet")["fraud_score"]
    else:
        train = pd.read_parquet(train_path, columns=FEATURE_COLUMNS)
        features = train.sample(
            min(mcfg.reference_size, len(train)), random_state=mcfg.random_state
        ).reset_index(drop=True)
        validation = pd.read_parquet(val_path, columns=FEATURE_COLUMNS)
        scores = pd.Series(detector.predict_proba(validation), name="fraud_score")
        info = {
            "key": key,
            "train_low": train.min().to_dict(),
            "train_high": train.max().to_dict(),
            "expected_alert_rate": float((scores >= detector.threshold).mean()),
        }
        cache.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache / "features.parquet", index=False)
        scores.to_frame().to_parquet(cache / "validation_scores.parquet", index=False)
        info_path.write_text(json.dumps(info, indent=2))
    return Reference(
        features, scores, info["expected_alert_rate"], info["train_low"], info["train_high"], info
    )
