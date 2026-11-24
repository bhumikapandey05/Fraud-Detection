"""Download the Kaggle dataset, or explain how to do it manually.

Kaggle requires a (free) account and an API token to download files, so this
module never fabricates data: if the download cannot be done automatically it
prints step-by-step instructions instead.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)

KAGGLE_SLUG = "mlg-ulb/creditcardfraud"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
# Byte-identical public copy of the same Kaggle file, hosted by TensorFlow for its
# imbalanced-data tutorial. Useful when you do not have Kaggle API credentials.
MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
FILE_NAME = "creditcard.csv"


def download_instructions(target_dir: Path) -> str:
    """Human-readable instructions for getting the dataset."""
    return f"""How to get the dataset (free):

  Option A - Kaggle website (no setup):
    1. Sign in at {KAGGLE_URL}
    2. Click "Download" and unzip the archive
    3. Move {FILE_NAME} to: {target_dir / FILE_NAME}

  Option B - Kaggle API:
    1. pip install kaggle
    2. Kaggle -> Settings -> API -> "Create New Token" (saves kaggle.json)
    3. mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    4. fraud-detection download --source kaggle

  Option C - public mirror of the same file (no account needed):
    fraud-detection download --source mirror
"""


def download_from_kaggle(target_dir: Path) -> Path:
    """Download and unzip the dataset using the official Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run `pip install kaggle`.\n\n"
            + download_instructions(target_dir)
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # the kaggle client raises generic errors for missing creds
        raise RuntimeError(
            f"Kaggle authentication failed: {exc}\n\n{download_instructions(target_dir)}"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from Kaggle into %s", KAGGLE_SLUG, target_dir)
    api.dataset_download_files(KAGGLE_SLUG, path=str(target_dir), unzip=True, quiet=False)
    return target_dir / FILE_NAME


def download_from_mirror(target_dir: Path, url: str = MIRROR_URL) -> Path:
    """Download the CSV from the public mirror (streams to disk, ~150 MB)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / FILE_NAME
    tmp = destination.with_suffix(".csv.part")
    logger.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(response, fh)
    tmp.replace(destination)  # only expose the file once it is complete
    return destination


def download_dataset(target_dir: Path, source: str = "kaggle", force: bool = False) -> Path:
    """Fetch the dataset into ``target_dir`` unless it already exists."""
    destination = target_dir / FILE_NAME
    if destination.is_file() and not force:
        logger.info("Dataset already present at %s (use --force to re-download)", destination)
        return destination

    if source == "kaggle":
        path = download_from_kaggle(target_dir)
    elif source == "mirror":
        path = download_from_mirror(target_dir)
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'kaggle' or 'mirror'")

    if not path.is_file():
        raise RuntimeError(f"Download finished but {path} was not found.")
    logger.info("Dataset ready at %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path
