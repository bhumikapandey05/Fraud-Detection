"""Comparison tables, charts and the Markdown evaluation report for Phase 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from fraud_detection.analysis.style import (
    COLOR_FRAUD,
    COLOR_LEGIT,
    SEQUENTIAL_CMAP,
    SERIES_COLORS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THOUSANDS,
    plt,
    save_figure,
)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """One row per model with validation metrics at the tuned threshold.

    Each input row needs: ``model``, ``val`` (metrics dict at tuned threshold),
    ``val_default`` (metrics at 0.5), ``val_ci`` and ``train_seconds``.
    """
    records = []
    for r in rows:
        val, ci = r["val"], r.get("val_ci", {})
        pr_ci = ci.get("pr_auc", [np.nan, np.nan])
        records.append(
            {
                "model": r["model"],
                "selected": r.get("selected", False),
                "val_pr_auc": val["pr_auc"],
                "val_pr_auc_ci_low": pr_ci[0],
                "val_pr_auc_ci_high": pr_ci[1],
                "val_roc_auc": val["roc_auc"],
                "threshold": val["threshold"],
                "precision": val["precision"],
                "recall": val["recall"],
                "f1": val["f1"],
                "tp": val["tp"],
                "fp": val["fp"],
                "fn": val["fn"],
                "expected_cost": val.get("expected_cost", np.nan),
                "precision_at_default": r["val_default"]["precision"],
                "recall_at_default": r["val_default"]["recall"],
                "fp_at_default": r["val_default"]["fp"],
                "train_seconds": r["train_seconds"],
            }
        )
    return pd.DataFrame(records)


def _fmt_ci(value: float, low: float, high: float) -> str:
    if np.isnan(low):
        return f"{value:.3f}"
    return f"{value:.3f} ({low:.3f}–{high:.3f})"


def comparison_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Model | PR-AUC (95% CI) | ROC-AUC | Threshold | Precision | Recall | F1 "
        "| Caught | False alarms | Missed | Cost ($) | Train (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        name = f"**{r.model}** ✓" if r.selected else r.model
        lines.append(
            f"| {name} | {_fmt_ci(r.val_pr_auc, r.val_pr_auc_ci_low, r.val_pr_auc_ci_high)} "
            f"| {r.val_roc_auc:.3f} | {r.threshold:.3f} | {r.precision:.3f} | {r.recall:.3f} "
            f"| {r.f1:.3f} | {r.tp} | {r.fp} | {r.fn} | {r.expected_cost:,.0f} "
            f"| {r.train_seconds:.1f} |"
        )
    return "\n".join(lines)


def threshold_effect_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Model | Precision @0.5 | Recall @0.5 | False alarms @0.5 "
        "| Precision @tuned | Recall @tuned | False alarms @tuned |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| {r.model} | {r.precision_at_default:.3f} | {r.recall_at_default:.3f} "
            f"| {r.fp_at_default} | {r.precision:.3f} | {r.recall:.3f} | {r.fp} |"
        )
    return "\n".join(lines)


def dataframe_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---:" for _ in df.columns) + "|"
    body = [
        "| " + " | ".join(f"{v:,}" if isinstance(v, (int, float)) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *body])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_pr_curves(curves: dict[str, dict], out: Path, title: str) -> Path:
    """Precision-recall curves. ``curves[label] = {"y", "scores", "threshold"}``."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    prevalence = None
    for color, (label, c) in zip(SERIES_COLORS, curves.items(), strict=False):
        precision, recall, _ = precision_recall_curve(c["y"], c["scores"])
        ax.step(recall, precision, where="post", color=color, linewidth=2, label=label)
        # Mark the operating point chosen on validation data.
        pred = np.asarray(c["scores"]) >= c["threshold"]
        y = np.asarray(c["y"]).astype(bool)
        tp = (pred & y).sum()
        if pred.sum():
            ax.plot(
                tp / y.sum(),
                tp / pred.sum(),
                "o",
                markersize=9,
                color=color,
                markeredgecolor="white",
                markeredgewidth=2,
                zorder=5,
            )
        prevalence = y.mean()
    if prevalence is not None:
        ax.axhline(prevalence, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(
            0.5,
            prevalence + 0.02,
            f"random model (fraud rate {prevalence:.2%})",
            color=TEXT_SECONDARY,
            fontsize=8,
            ha="center",
        )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title(title)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 0.06), title="● = tuned threshold", title_fontsize=8
    )
    return save_figure(fig, out)


def plot_threshold_tradeoff(curve: pd.DataFrame, threshold: float, out: Path, title: str) -> Path:
    """Precision/recall vs threshold (top) and expected cost vs threshold (bottom)."""
    curve = curve.sort_values("threshold")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    top.plot(
        curve["threshold"], curve["precision"], color=COLOR_LEGIT, linewidth=2, label="Precision"
    )
    top.plot(curve["threshold"], curve["recall"], color=COLOR_FRAUD, linewidth=2, label="Recall")
    top.set_ylim(0, 1.03)
    top.set_ylabel("Score")
    top.set_title(title)
    top.legend(loc="lower center", ncols=2)

    bottom.plot(curve["threshold"], curve["expected_cost"], color=TEXT_PRIMARY, linewidth=2)
    best_cost = curve.loc[(curve["threshold"] - threshold).abs().idxmin(), "expected_cost"]
    bottom.plot(
        threshold,
        best_cost,
        "o",
        markersize=9,
        color=TEXT_PRIMARY,
        markeredgecolor="white",
        markeredgewidth=2,
        zorder=5,
    )
    bottom.set_ylabel("Expected cost ($, axis capped)")
    bottom.yaxis.set_major_formatter(THOUSANDS)
    bottom.set_xlabel("Decision threshold (flag if fraud score ≥ threshold)")
    # Flagging almost everything costs orders of magnitude more; cap the axis at 1.5x the
    # cost at the highest threshold so the region around the optimum stays visible.
    bottom.set_ylim(0, 1.5 * curve["expected_cost"].iloc[-1])
    for ax in (top, bottom):
        ax.axvline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    bottom.annotate(
        f"chosen = {threshold:.3f}",
        (threshold, best_cost),
        xytext=(10, 25),
        textcoords="offset points",
        color=TEXT_PRIMARY,
    )
    return save_figure(fig, out)


def plot_confusion_matrices(matrices: dict[str, dict], out: Path) -> Path:
    """Annotated confusion matrices; ``matrices[label] = {"tn","fp","fn","tp"}``."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.9))
    for ax, (label, c) in zip(np.atleast_1d(axes), matrices.items(), strict=False):
        grid = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        # Colour each row by its own share so the tiny fraud row stays readable.
        shares = grid / grid.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
        names = [["True negative", "False alarm"], ["Missed fraud", "Fraud caught"]]
        for i in range(2):
            for j in range(2):
                ink = "white" if shares[i, j] > 0.55 else TEXT_PRIMARY
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:,}\n{names[i][j]}\n({shares[i, j]:.1%} of row)",
                    ha="center",
                    va="center",
                    color=ink,
                    fontsize=9,
                )
        ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
        ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
        ax.grid(False)
        ax.set_title(label, fontsize=11)
    return save_figure(fig, out)


def write_markdown(path: Path, sections: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections).rstrip() + "\n")
    return path
