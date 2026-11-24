"""Exploratory data analysis (EDA) as reusable, testable functions.

``compute_eda_summary`` returns plain numbers (saved as JSON so they can be
diffed between data versions); ``generate_eda_figures`` saves PNG charts;
``write_eda_report`` combines both into a readable Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.analysis.style import (
    COLOR_FRAUD,
    COLOR_LEGIT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THOUSANDS,
    plt,
    save_figure,
)
from fraud_detection.data.schema import (
    AMOUNT_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from fraud_detection.logging_utils import get_logger

logger = get_logger(__name__)

CLASS_LABELS = {0: "Legitimate", 1: "Fraud"}
CLASS_COLORS = {0: COLOR_LEGIT, 1: COLOR_FRAUD}


# --------------------------------------------------------------------------- #
# Numeric summary
# --------------------------------------------------------------------------- #
def _relative_hour(time_seconds: pd.Series) -> pd.Series:
    """Hour of day (0-23) relative to the first transaction."""
    return ((time_seconds // 3600) % 24).astype(int)


def feature_target_correlation(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.Series:
    """Pearson correlation of each feature with the 0/1 label (point-biserial)."""
    corr = df[FEATURE_COLUMNS].corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def compute_eda_summary(df: pd.DataFrame, target: str = TARGET_COLUMN) -> dict:
    """Compute the key EDA statistics as a JSON-serialisable dict."""
    counts = df[target].value_counts().sort_index()
    n = len(df)
    fraud = int(counts.get(1, 0))
    legit = int(counts.get(0, 0))

    dup_mask = df.duplicated(keep="first")
    amount_by_class = (
        df.groupby(target)[AMOUNT_COLUMN]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        .round(2)
    )
    hours = _relative_hour(df[TIME_COLUMN])
    hourly = df.groupby(hours)[target].agg(["size", "mean"])
    corr = feature_target_correlation(df, target)

    return {
        "shape": {"rows": n, "columns": df.shape[1]},
        "class_distribution": {
            "legitimate": legit,
            "fraud": fraud,
            "fraud_rate_pct": round(100 * fraud / n, 4) if n else 0.0,
            "imbalance_ratio": round(legit / fraud, 1) if fraud else None,
        },
        "missing_values": {
            "total": int(df.isna().sum().sum()),
            "by_column": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        },
        "duplicates": {
            "duplicate_rows": int(dup_mask.sum()),
            "fraud_duplicate_rows": int(df.loc[dup_mask, target].sum()),
        },
        "amount": {
            "overall": df[AMOUNT_COLUMN].describe().round(2).to_dict(),
            "by_class": {
                CLASS_LABELS[int(k)]: v for k, v in amount_by_class.to_dict("index").items()
            },
            "zero_amount_transactions": int((df[AMOUNT_COLUMN] == 0).sum()),
        },
        "time": {
            "span_hours": round(float(df[TIME_COLUMN].max() - df[TIME_COLUMN].min()) / 3600, 2),
            "transactions_per_relative_hour": {int(h): int(v) for h, v in hourly["size"].items()},
            "fraud_rate_pct_per_relative_hour": {
                int(h): round(100 * float(v), 4) for h, v in hourly["mean"].items()
            },
        },
        "top_correlated_features": {k: round(float(v), 4) for k, v in corr.head(10).items()},
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_class_distribution(df: pd.DataFrame, out: Path, target: str = TARGET_COLUMN) -> Path:
    counts = df[target].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    labels = [CLASS_LABELS[c] for c in counts.index]
    ax.barh(labels, counts.values, color=[CLASS_COLORS[c] for c in counts.index], height=0.55)
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(
            v + total * 0.01, i, f"{v:,}  ({100 * v / total:.3f}%)", va="center", color=TEXT_PRIMARY
        )
    ax.set_xlim(0, total * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Transactions")
    ax.set_title("Class distribution: fraud is a tiny minority")
    return save_figure(fig, out)


def plot_amount_distribution(df: pd.DataFrame, out: Path, target: str = TARGET_COLUMN) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    log_amount = np.log1p(df[AMOUNT_COLUMN])
    bins = np.linspace(0, log_amount.max(), 60)
    for cls in (0, 1):
        ax.hist(
            log_amount[df[target] == cls],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=CLASS_COLORS[cls],
            label=CLASS_LABELS[cls],
        )
    ticks = [0, 1, 10, 100, 1_000, 10_000]
    ax.set_xticks(np.log1p(ticks), [f"{t:,}" for t in ticks])
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Density (each class sums to 1)")
    ax.set_title("Transaction amount by class")
    ax.legend()
    return save_figure(fig, out)


def plot_hourly_pattern(df: pd.DataFrame, out: Path, target: str = TARGET_COLUMN) -> Path:
    hours = _relative_hour(df[TIME_COLUMN])
    hourly = df.groupby(hours)[target].agg(["size", "mean"]).reindex(range(24), fill_value=0)
    # Two measures with different units -> two panels sharing the x-axis (no dual axis).
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    top.bar(hourly.index, hourly["size"], color=COLOR_LEGIT, width=0.8)
    top.yaxis.set_major_formatter(THOUSANDS)
    top.set_ylabel("Transactions")
    top.set_title("Volume and fraud rate by hour (relative to first transaction)")
    top.grid(axis="x", visible=False)
    bottom.plot(
        hourly.index, 100 * hourly["mean"], color=COLOR_FRAUD, linewidth=2, marker="o", markersize=5
    )
    bottom.set_ylabel("Fraud rate (%)")
    bottom.set_xlabel("Hour of day (relative)")
    bottom.set_xticks(range(0, 24, 2))
    bottom.set_ylim(bottom=0)
    return save_figure(fig, out)


def plot_feature_correlation(df: pd.DataFrame, out: Path, target: str = TARGET_COLUMN) -> Path:
    corr = feature_target_correlation(df, target).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.barh(corr.index, corr.values, color=COLOR_LEGIT, height=0.7)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Correlation with fraud label")
    ax.set_title("Feature correlation with Class (sorted by strength)")
    ax.tick_params(axis="y", labelsize=8)
    return save_figure(fig, out)


def plot_top_feature_distributions(
    df: pd.DataFrame, out: Path, target: str = TARGET_COLUMN, n_features: int = 6
) -> Path:
    top = feature_target_correlation(df, target).index[:n_features]
    ncols = 3
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    for ax, col in zip(np.atleast_1d(axes).ravel(), top, strict=False):
        # Range covers the bulk of BOTH classes; extreme outliers fall outside the bins
        # (they are dropped from the plot, not clipped, so no artificial edge spikes).
        legit, fraud = df.loc[df[target] == 0, col], df.loc[df[target] == 1, col]
        lo = min(legit.quantile(0.001), fraud.quantile(0.02))
        hi = max(legit.quantile(0.999), fraud.quantile(0.98))
        bins = np.linspace(lo, hi, 50)
        for cls in (0, 1):
            ax.hist(
                df.loc[df[target] == cls, col],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=CLASS_COLORS[cls],
                label=CLASS_LABELS[cls],
            )
        ax.set_title(col, fontsize=10)
        ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[len(top) :]:
        ax.set_visible(False)
    np.atleast_1d(axes).ravel()[0].legend(loc="upper left")
    fig.suptitle(
        "Most discriminative features: fraud vs legitimate distributions",
        x=0.01,
        ha="left",
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    return save_figure(fig, out)


def generate_eda_figures(
    df: pd.DataFrame, out_dir: Path, target: str = TARGET_COLUMN
) -> list[Path]:
    """Create all EDA charts in ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_class_distribution(df, out_dir / "class_distribution.png", target),
        plot_amount_distribution(df, out_dir / "amount_distribution.png", target),
        plot_hourly_pattern(df, out_dir / "hourly_pattern.png", target),
        plot_feature_correlation(df, out_dir / "feature_correlation.png", target),
        plot_top_feature_distributions(df, out_dir / "top_feature_distributions.png", target),
    ]
    logger.info("Saved %d figures to %s", len(figures), out_dir)
    return figures


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_eda_report(summary: dict, figures: list[Path], out_dir: Path) -> Path:
    """Write ``eda_summary.json`` and a human-readable ``eda_report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    cd = summary["class_distribution"]
    amt = summary["amount"]["by_class"]
    dup = summary["duplicates"]
    top = summary["top_correlated_features"]
    lines = [
        "# EDA Report - Credit Card Fraud Detection",
        "",
        f"- Rows: **{summary['shape']['rows']:,}**, columns: **{summary['shape']['columns']}**",
        f"- Fraud cases: **{cd['fraud']:,}** ({cd['fraud_rate_pct']}%), "
        f"about **1 in {cd['imbalance_ratio']:,.0f}** transactions",
        f"- Missing values: **{summary['missing_values']['total']}**",
        f"- Exact duplicate rows: **{dup['duplicate_rows']:,}** "
        f"({dup['fraud_duplicate_rows']} of them fraud)",
        f"- Time span: **{summary['time']['span_hours']} hours** (~2 days)",
        f"- Zero-amount transactions: **{summary['amount']['zero_amount_transactions']:,}**",
        "",
        "## Amount by class",
        "",
        "| Class | mean | median | 95th pct | 99th pct | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in amt.items():
        cells = [f"{s[key]:,}" for key in ("mean", "50%", "95%", "99%", "max")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Features most correlated with fraud", "", "| Feature | corr |", "|---|---:|"]
    lines += [f"| {k} | {v:+.3f} |" for k, v in top.items()]
    lines += ["", "## Figures", ""]
    lines += [f"![{p.stem}](figures/{p.name})" for p in figures]
    report = out_dir / "eda_report.md"
    report.write_text("\n".join(lines) + "\n")
    logger.info("EDA report written to %s", report)
    return report
