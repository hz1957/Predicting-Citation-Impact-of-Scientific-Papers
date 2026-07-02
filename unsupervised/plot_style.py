"""Shared matplotlib style for the unsupervised-learning figures.

Palette follows the validated categorical/sequential scheme used across the
project figures (colorblind-checked with Machado-2009 simulation; the three
sub-3:1-contrast hues are always paired with direct labels).
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, fixed order — cluster i always wears CATEGORICAL[i].
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]

# Sequential single-hue blue ramp (steps 100 -> 700), light = low.
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 110,
    })


def save_fig(fig, name, out_web, out_paper):
    """Save a figure as web PNG (2x) and paper PDF."""
    fig.savefig(out_web / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_paper / f"{name}.pdf", bbox_inches="tight")
