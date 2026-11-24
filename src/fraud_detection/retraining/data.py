"""Assemble the data a candidate may be trained on - and nothing else.

Permitted for TRAINING:
* the original training split (verified byte-identical to the champion's), and
* labelled production transactions from the monitoring store, except the newest
  ``holdout_fraction`` of them.

Kept out of training:
* the newest labelled production rows -> a **recent holdout** for comparison;
* the validation split -> the fixed yardstick for comparison (as in Phase 2);
* the **test split** -> never read at all. Every file goes through
  ``read_permitted``, which refuses the test split by name and by checksum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fraud_detection.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_detection.monitoring.store import MonitoringStore
from fraud_detection.pipelines.train import _file_sha256


class ForbiddenDataError(RuntimeError):
    """An attempt was made to use data that retraining must never touch."""


def read_permitted(path: Path, forbidden_sha256: set[str], purpose: str) -> pd.DataFrame:
    """Read a split file after checking it is not a forbidden (e.g. test) split."""
    path = Path(path)
    if path.name == "test.parquet":
        raise ForbiddenDataError(f"Refusing to read the test split for {purpose}: {path}")
    if _file_sha256(path) in forbidden_sha256:
        raise ForbiddenDataError(f"{path} has the checksum of a forbidden split ({purpose})")
    return pd.read_parquet(path)


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of a DataFrame's contents (reproducibility record)."""
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


@dataclass
class RetrainingData:
    train: pd.DataFrame  # features + Class, used to fit the candidate
    validation: pd.DataFrame  # comparison yardstick (never trained on)
    holdout: pd.DataFrame  # newest labelled production rows (never trained on)
    manifest: dict = field(default_factory=dict)


def assemble_training_data(
    processed_dir: Path,
    store: MonitoringStore,
    champion_manifest: dict,
    use_production_labels: bool,
    holdout_fraction: float,
) -> RetrainingData:
    split_hashes = champion_manifest.get("split_files_sha256", {})
    forbidden = {split_hashes["test"]} if "test" in split_hashes else set()

    train_path = Path(processed_dir) / "train.parquet"
    base = read_permitted(train_path, forbidden, "candidate training")
    train_sha = _file_sha256(train_path)
    if "train" in split_hashes and train_sha != split_hashes["train"]:
        raise ForbiddenDataError("train.parquet differs from the champion's training split")
    validation = read_permitted(
        Path(processed_dir) / "validation.parquet", forbidden | {train_sha}, "candidate comparison"
    )

    labelled = store.predictions(labeled_only=True) if use_production_labels else pd.DataFrame()
    if len(labelled):
        labelled = (
            labelled.drop_duplicates("transaction_id", keep="last")
            .sort_values("id")
            .reset_index(drop=True)
        )
        labelled[TARGET_COLUMN] = labelled["label"].astype(int)
    n_holdout = int(round(len(labelled) * holdout_fraction))
    production_train = labelled.iloc[: len(labelled) - n_holdout]
    holdout = labelled.iloc[len(labelled) - n_holdout :]

    columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    train = pd.concat(
        [base[columns], production_train[columns]] if len(production_train) else [base[columns]],
        ignore_index=True,
    )
    manifest = {
        "sources": {
            "train_split": {
                "rows": len(base),
                "frauds": int(base[TARGET_COLUMN].sum()),
                "sha256": train_sha,
            },
            "production_labels": {
                "rows": len(production_train),
                "frauds": int(production_train[TARGET_COLUMN].sum())
                if len(production_train)
                else 0,
                "batches": sorted(production_train["batch_id"].unique().tolist())
                if len(production_train)
                else [],
            },
        },
        "holdout": {
            "rows": len(holdout),
            "frauds": int(holdout[TARGET_COLUMN].sum()) if len(holdout) else 0,
            "batches": sorted(holdout["batch_id"].unique().tolist()) if len(holdout) else [],
        },
        "validation_sha256": _file_sha256(Path(processed_dir) / "validation.parquet"),
        "test_split": "never read (refused by name and checksum)",
        "training_rows": len(train),
        "training_frauds": int(train[TARGET_COLUMN].sum()),
        "training_fingerprint": frame_fingerprint(train),
    }
    return RetrainingData(
        train,
        validation[columns],
        holdout[columns + ["batch_id"]] if len(holdout) else pd.DataFrame(columns=columns),
        manifest,
    )
