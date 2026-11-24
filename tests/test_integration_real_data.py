"""Checks against the real Kaggle file. Skipped automatically if it is not downloaded."""

import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.data.cleaning import clean_data
from fraud_detection.data.loader import load_raw_data
from fraud_detection.data.split import split_data
from fraud_detection.data.validation import validate_raw_data, validate_splits

from .conftest import REAL_DATA_PATH

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REAL_DATA_PATH.is_file(), reason="Kaggle dataset not downloaded"),
]


@pytest.fixture(scope="module")
def real_df():
    return load_raw_data(REAL_DATA_PATH)


def test_real_data_matches_published_facts(real_df):
    assert real_df.shape == (284_807, 31)
    assert int(real_df["Class"].sum()) == 492
    assert validate_raw_data(real_df, expected_rows=284_807).passed


def test_real_data_time_split_is_leakage_safe(real_df):
    clean, summary = clean_data(real_df)
    assert summary.duplicates_removed == 1_081
    cfg = SplitConfig()
    splits = split_data(clean, cfg)
    report = validate_splits(
        splits.as_dict(), strategy="time", min_positives=cfg.min_positives_per_split
    )
    assert report.passed, report.errors
