"""Load the raw CSV and persist / reload intermediate tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.data.download import download_instructions
from fraud_detection.data.schema import COLUMN_DTYPES, EXPECTED_COLUMNS
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw Kaggle CSV has not been downloaded yet."""


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read ``creditcard.csv`` with explicit dtypes.

    Raises:
        DatasetNotFoundError: if the file does not exist. The error message
            explains how to download it. No data is ever generated as a fallback.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}.\n\n{download_instructions(path.parent)}"
        )

    logger.info("Loading raw data from %s", path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {missing}")

    df = pd.read_csv(path, dtype=COLUMN_DTYPES)
    logger.info("Loaded %s rows x %s columns", f"{len(df):,}", df.shape[1])
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame as Parquet (fast, compact, keeps dtypes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s rows to %s", f"{len(df):,}", path)
    return path


def load_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `fraud-detection prepare` first.")
    return pd.read_parquet(path)
