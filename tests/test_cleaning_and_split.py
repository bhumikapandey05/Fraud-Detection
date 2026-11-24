import pandas as pd
import pytest

from fraud_detection.config import SplitConfig
from fraud_detection.data.cleaning import clean_data
from fraud_detection.data.split import split_data, stratified_split, time_based_split


def test_clean_data_removes_duplicates_and_sorts(synthetic_df):
    dupes = synthetic_df[synthetic_df["Class"] == 1].head(2)
    shuffled = pd.concat([synthetic_df, dupes]).sample(frac=1, random_state=0)
    clean, summary = clean_data(shuffled)
    assert summary.duplicates_removed == 2
    assert summary.fraud_duplicates_removed == 2
    assert len(clean) == len(synthetic_df)
    assert clean["Time"].is_monotonic_increasing
    assert not clean.duplicated().any()


def test_clean_data_can_keep_duplicates(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(1)])
    clean, summary = clean_data(df, drop_duplicates=False)
    assert summary.duplicates_removed == 0
    assert len(clean) == len(df)


def test_time_split_is_strictly_chronological(synthetic_df):
    splits = time_based_split(synthetic_df, 0.7, 0.15)
    assert splits.train["Time"].max() < splits.validation["Time"].min()
    assert splits.validation["Time"].max() < splits.test["Time"].min()
    assert sum(len(d) for d in splits.as_dict().values()) == len(synthetic_df)


def test_time_split_never_splits_a_shared_timestamp():
    # 10 rows share one timestamp right at the 70% boundary.
    df = pd.DataFrame({"Time": [0] * 65 + [5] * 10 + list(range(10, 35)), "Class": 0})
    splits = time_based_split(df, 0.7, 0.15)
    for name, part in splits.as_dict().items():
        others = pd.concat([p for n, p in splits.as_dict().items() if n != name])
        assert not set(part["Time"]) & set(others["Time"])


def test_time_split_sizes_are_close_to_config(synthetic_df):
    splits = time_based_split(synthetic_df, 0.7, 0.15)
    n = len(synthetic_df)
    assert len(splits.train) / n == pytest.approx(0.70, abs=0.01)
    assert len(splits.validation) / n == pytest.approx(0.15, abs=0.01)


def test_time_split_requires_rows():
    with pytest.raises(ValueError):
        time_based_split(pd.DataFrame({"Time": [1, 2], "Class": [0, 1]}), 0.7, 0.15)


def test_stratified_split_preserves_fraud_rate(synthetic_df):
    splits = stratified_split(synthetic_df, 0.7, 0.15, random_state=1)
    overall = synthetic_df["Class"].mean()
    for part in splits.as_dict().values():
        assert part["Class"].mean() == pytest.approx(overall, abs=0.01)


def test_split_data_dispatches_on_strategy(synthetic_df):
    time_splits = split_data(synthetic_df, SplitConfig(strategy="time"))
    assert time_splits.train["Time"].max() < time_splits.test["Time"].min()
    strat = split_data(synthetic_df, SplitConfig(strategy="stratified"))
    summary = strat.summary()
    assert set(summary) == {"train", "validation", "test"}
    assert summary["train"]["rows"] + summary["validation"]["rows"] + summary["test"][
        "rows"
    ] == len(synthetic_df)
