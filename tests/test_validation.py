import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.validation import (
    DataValidationError,
    validate_raw_data,
    validate_splits,
)


def test_clean_data_passes(synthetic_df):
    report = validate_raw_data(synthetic_df)
    assert report.passed, report.errors
    assert report.checks_run > 5


def test_expected_row_count_mismatch_is_only_a_warning(synthetic_df):
    report = validate_raw_data(synthetic_df, expected_rows=284807)
    assert report.passed
    assert any("Row count" in w for w in report.warnings)


def test_missing_column_fails(synthetic_df):
    report = validate_raw_data(synthetic_df.drop(columns=["V3"]))
    assert not report.passed
    with pytest.raises(DataValidationError, match="V3"):
        report.raise_if_failed()


def test_missing_values_fail(synthetic_df):
    df = synthetic_df.copy()
    df.loc[5, "V1"] = np.nan
    report = validate_raw_data(df)
    assert any("Missing values" in e for e in report.errors)


def test_infinite_values_fail(synthetic_df):
    df = synthetic_df.copy()
    df.loc[5, "Amount"] = np.inf
    assert any("Infinite" in e for e in validate_raw_data(df).errors)


def test_invalid_labels_fail(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Class"] = 2
    assert any("invalid values" in e for e in validate_raw_data(df).errors)


def test_single_class_fails(synthetic_df):
    df = synthetic_df.assign(Class=0)
    assert any("both classes" in e for e in validate_raw_data(df).errors)


def test_negative_amount_fails(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "Amount"] = -1.0
    assert any("negative" in e for e in validate_raw_data(df).errors)


def test_non_numeric_column_fails(synthetic_df):
    df = synthetic_df.astype({"V2": "object"})
    df.loc[0, "V2"] = "oops"
    assert any("Non-numeric" in e for e in validate_raw_data(df).errors)


def test_duplicates_are_a_warning(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(3)], ignore_index=True)
    report = validate_raw_data(df)
    assert report.passed
    assert any("duplicate" in w for w in report.warnings)


# --- Split validation --------------------------------------------------------
def _time_splits(df):
    n = len(df)
    return {
        "train": df.iloc[: int(n * 0.7)],
        "validation": df.iloc[int(n * 0.7) : int(n * 0.85)],
        "test": df.iloc[int(n * 0.85) :],
    }


def test_valid_splits_pass(synthetic_df):
    df = synthetic_df.drop_duplicates(subset="Time")
    report = validate_splits(_time_splits(df), strategy="time", min_positives=5)
    assert report.passed, report.errors


def test_overlapping_rows_detected(synthetic_df):
    splits = _time_splits(synthetic_df)
    splits["test"] = pd.concat([splits["test"], splits["train"].head(2)])
    report = validate_splits(splits, strategy="stratified", min_positives=5)
    assert any("identical rows" in e for e in report.errors)


def test_time_leakage_detected(synthetic_df):
    splits = _time_splits(synthetic_df)
    splits["train"], splits["test"] = splits["test"], splits["train"]
    report = validate_splits(splits, strategy="time", min_positives=5)
    assert any("Time leakage" in e for e in report.errors)


def test_too_few_positives_detected(synthetic_df):
    report = validate_splits(_time_splits(synthetic_df), strategy="time", min_positives=10_000)
    assert any("fraud cases" in e for e in report.errors)
