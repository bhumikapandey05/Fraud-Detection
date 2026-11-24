import sys
from pathlib import Path

import pandas as pd
import pytest

from fraud_detection.data.download import (
    KAGGLE_URL,
    download_dataset,
    download_from_kaggle,
    download_from_mirror,
    download_instructions,
)
from fraud_detection.data.loader import (
    DatasetNotFoundError,
    load_parquet,
    load_raw_data,
    save_parquet,
)


def test_missing_dataset_explains_how_to_download(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_raw_data(tmp_path / "creditcard.csv")
    assert KAGGLE_URL in str(exc.value)


def test_load_raw_data_dtypes(tmp_path: Path, synthetic_df: pd.DataFrame):
    path = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(path, index=False)
    df = load_raw_data(path)
    assert df.shape == synthetic_df.shape
    assert df["Class"].dtype == "int64"
    assert df["V1"].dtype == "float64"


def test_load_raw_data_missing_column(tmp_path: Path, synthetic_df: pd.DataFrame):
    path = tmp_path / "creditcard.csv"
    synthetic_df.drop(columns=["Amount"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Amount"):
        load_raw_data(path)


def test_parquet_roundtrip(tmp_path: Path, synthetic_df: pd.DataFrame):
    path = save_parquet(synthetic_df, tmp_path / "sub" / "x.parquet")
    pd.testing.assert_frame_equal(load_parquet(path), synthetic_df)


def test_download_skips_when_file_exists(tmp_path: Path):
    existing = tmp_path / "creditcard.csv"
    existing.write_text("already here")
    assert download_dataset(tmp_path, source="mirror") == existing
    assert existing.read_text() == "already here"  # nothing was downloaded


def test_download_unknown_source(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown source"):
        download_dataset(tmp_path, source="ftp")


def test_instructions_mention_target_path(tmp_path: Path):
    assert str(tmp_path / "creditcard.csv") in download_instructions(tmp_path)


def test_mirror_download_writes_complete_file(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("Time,Class\n0,0\n")
    target_dir = tmp_path / "raw"
    path = download_from_mirror(target_dir, url=source.as_uri())  # local file:// URL, no network
    assert path.read_text() == source.read_text()
    assert not list(target_dir.glob("*.part"))


def test_kaggle_download_without_package_gives_instructions(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", None)  # simulate ImportError
    with pytest.raises(RuntimeError, match="pip install kaggle"):
        download_from_kaggle(tmp_path)
