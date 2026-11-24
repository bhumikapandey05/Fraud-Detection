"""Data preparation pipeline.

Steps (each one logged):

1. Load the raw Kaggle CSV.
2. Validate it (schema, types, missing values, labels, ranges). Stop on errors.
3. Clean it (drop exact duplicates, sort by time).
4. Split it into train / validation / test (time-based by default).
5. Validate the splits (no overlap, no time leakage, enough fraud cases).
6. Save the splits as Parquet.
7. Fit the preprocessor on the TRAINING split only, check it transforms all
   three splits cleanly, and save it for the modelling phase.
8. Write a JSON report describing everything that happened.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from fraud_detection.config import Config
from fraud_detection.data.cleaning import clean_data
from fraud_detection.data.loader import load_raw_data, save_parquet
from fraud_detection.data.split import split_data
from fraud_detection.data.validation import (
    DataValidationError,
    validate_raw_data,
    validate_splits,
)
from fraud_detection.features.preprocessing import build_preprocessor, split_features_target
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)

PREPROCESSOR_FILE = "preprocessor.joblib"
REPORT_FILE = "data_preparation_report.json"


def run_prepare(cfg: Config) -> dict:
    """Execute the full data preparation pipeline and return its report."""
    target, time_col = cfg.dataset.target, cfg.dataset.time_column

    # 1-2. Load and validate the raw data.
    raw = load_raw_data(cfg.paths.raw_data)
    raw_report = validate_raw_data(raw, expected_rows=cfg.dataset.expected_rows)
    raw_report.log()
    raw_report.raise_if_failed()

    # 3. Clean.
    clean, cleaning_summary = clean_data(
        raw, drop_duplicates=cfg.cleaning.drop_duplicates, target=target, time_column=time_col
    )
    save_parquet(clean, cfg.paths.interim_dir / "creditcard_clean.parquet")

    # 4-5. Split and validate the splits.
    splits = split_data(clean, cfg.split, target=target, time_column=time_col)
    split_report = validate_splits(
        splits.as_dict(),
        strategy=cfg.split.strategy,
        min_positives=cfg.split.min_positives_per_split,
        target=target,
        time_column=time_col,
    )
    split_report.log()
    split_report.raise_if_failed()

    # 6. Persist splits (untransformed, so later phases can rebuild features freely).
    split_paths = {
        name: str(save_parquet(df, cfg.paths.processed_dir / f"{name}.parquet"))
        for name, df in splits.as_dict().items()
    }

    # 7. Fit preprocessing on training data ONLY, then apply to every split.
    preprocessor = build_preprocessor(cfg.preprocessing)
    X_train, y_train = split_features_target(splits.train, target)
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()

    transformed_shapes = {}
    for name, df in splits.as_dict().items():
        X, _ = split_features_target(df, target)
        Xt = preprocessor.transform(X)
        if not np.isfinite(Xt.to_numpy()).all():
            raise DataValidationError(f"Preprocessed '{name}' split contains NaN/inf values")
        transformed_shapes[name] = list(Xt.shape)
    logger.info("Preprocessor fitted on train; output features: %d", len(feature_names))

    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = cfg.paths.artifacts_dir / PREPROCESSOR_FILE
    joblib.dump(preprocessor, preprocessor_path)
    logger.info("Saved fitted preprocessor to %s", preprocessor_path)

    # 8. Report.
    n_pos = int(y_train.sum())
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_data": str(cfg.paths.raw_data),
        "raw_validation": raw_report.to_dict(),
        "cleaning": cleaning_summary.to_dict(),
        "split_strategy": cfg.split.strategy,
        "splits": splits.summary(target, time_col),
        "split_validation": split_report.to_dict(),
        "split_files": split_paths,
        "preprocessor": {
            "path": str(preprocessor_path),
            "fitted_on": "train",
            "output_features": feature_names,
            "transformed_shapes": transformed_shapes,
        },
        # Handy for Phase 2: XGBoost's scale_pos_weight = negatives / positives in TRAIN.
        "train_class_ratio_neg_to_pos": round((len(y_train) - n_pos) / n_pos, 2),
    }
    report_path = cfg.paths.artifacts_dir / REPORT_FILE
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Data preparation report written to %s", report_path)
    return report


def load_preprocessor(artifacts_dir: Path):
    """Load the fitted preprocessor saved by :func:`run_prepare`."""
    path = Path(artifacts_dir) / PREPROCESSOR_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `fraud-detection prepare` first.")
    return joblib.load(path)
