"""Shared chart styling: one neutral surface, recessive axes, validated hues.

Importing this module selects the non-interactive Agg backend (charts are
written to files) and applies the project's matplotlib defaults.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Categorical hues in fixed order (first three are colour-blind-safe as a set).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_LEGIT, COLOR_FRAUD = SERIES_COLORS[0], SERIES_COLORS[1]
SEQUENTIAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#eef4fc", "#9ec5f4", "#2a78d6", "#104281"]
)

THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_PRIMARY,
        "font.size": 10,
    }
)


def save_figure(fig: plt.Figure, path) -> object:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
