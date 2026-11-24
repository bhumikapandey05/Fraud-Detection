"""EDA pipeline: load data -> compute statistics -> save figures and report."""

from __future__ import annotations

from pathlib import Path

from fraud_detection.analysis.eda import (
    compute_eda_summary,
    generate_eda_figures,
    write_eda_report,
)
from fraud_detection.config import Config
from fraud_detection.data.loader import load_parquet, load_raw_data
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)


def run_eda(cfg: Config, on: str = "raw") -> Path:
    """Run EDA on the raw dataset (default) or on the training split only.

    ``on="raw"`` describes the full file (needed to see duplicates and missing
    values). ``on="train"`` restricts the analysis to the training split, which
    is the stricter choice when EDA findings are used to make modelling
    decisions, because it never looks at validation/test data.
    """
    if on == "raw":
        df = load_raw_data(cfg.paths.raw_data)
        out_dir = cfg.paths.eda_dir
    elif on == "train":
        df = load_parquet(cfg.paths.processed_dir / "train.parquet")
        out_dir = cfg.paths.eda_dir / "train"
    else:
        raise ValueError(f"on must be 'raw' or 'train', got {on!r}")

    summary = compute_eda_summary(df, cfg.dataset.target)
    figures = generate_eda_figures(df, out_dir / "figures", cfg.dataset.target)
    return write_eda_report(summary, figures, out_dir)
