"""Shared visual style for paper/generate_figures.py.

A single import point so all six paper figures look like one paper.

Conventions
-----------
* Colorblind-safe palette (Wong 2011) — no rainbows.
* Heatmaps use ``cividis`` (perceptually ordered + colorblind-safe).
* Embedded TrueType fonts (``pdf.fonttype=42``, ``ps.fonttype=42``) so
  PDF submissions pass strict camera-ready checks.
* Two width conventions:
    - ``WIDTH_SINGLE``: ~3.3 in (one ACL column)
    - ``WIDTH_DOUBLE``: ~7.2 in (full text width, two-column figures)
* Semantic color encoding is *consistent across figures*: the same
  concept always uses the same color (see :data:`SEMANTIC` table).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Wong 2011 colorblind-safe palette
# ---------------------------------------------------------------------------
WONG_BLUE = "#0072B2"        # blue
WONG_VERMILLION = "#D55E00"  # orange-red
WONG_GREEN = "#009E73"       # bluish-green
WONG_PURPLE = "#CC79A7"      # reddish-purple
WONG_ORANGE = "#E69F00"      # orange
WONG_SKY = "#56B4E9"         # sky blue
WONG_YELLOW = "#F0E442"      # yellow
WONG_BLACK = "#000000"

GRAY_DARK = "#555555"
GRAY_MID = "#999999"
GRAY_LIGHT = "#BBBBBB"
GRAY_BG = "#EEEEEE"


# ---------------------------------------------------------------------------
# Semantic color map — same concept, same color, every figure.
# ---------------------------------------------------------------------------
SEMANTIC = {
    # Class / direction
    "A": WONG_BLUE,            # answerable class
    "U": WONG_VERMILLION,      # unanswerable class
    "d_imp": WONG_PURPLE,      # impossibility direction
    "d_ref": WONG_GREEN,       # safety refusal direction
    # Subspaces (Fig 6)
    "null": WONG_BLUE,
    "pc": WONG_ORANGE,
    "full": GRAY_MID,
    # Pretraining contrast (Fig 3)
    "instruct": WONG_BLUE,
    "base": WONG_PURPLE,
    # Steering controls (Fig 5)
    "signal": WONG_BLUE,
    "random": GRAY_LIGHT,
    # Math/code/fact dataset accents (Fig 5 / Fig 2 axis labels)
    "math": WONG_BLUE,
    "code": WONG_VERMILLION,
    "fact": WONG_PURPLE,
}

# Heatmap colormap — perceptually ordered + colorblind-safe.
HEATMAP_CMAP = plt.cm.cividis


# ---------------------------------------------------------------------------
# Sizing — ACL/EMNLP column widths
# ---------------------------------------------------------------------------
WIDTH_SINGLE = 3.3   # one column
WIDTH_DOUBLE = 7.2   # full text width (two columns)


# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
SIZE_AXIS = 9
SIZE_TICK = 8
SIZE_LEGEND = 7.5
SIZE_TITLE = 9.5
SIZE_ANNOT = 7.5


# ---------------------------------------------------------------------------
# rcParams setup. Call :func:`apply` once at script start.
# ---------------------------------------------------------------------------
def apply() -> None:
    """Install global rcParams for all paper figures."""
    mpl.rcParams.update(
        {
            # Fonts
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": SIZE_AXIS,
            "axes.labelsize": SIZE_AXIS,
            "axes.titlesize": SIZE_TITLE,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "legend.fontsize": SIZE_LEGEND,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Lines
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            # Grids
            "grid.color": "#DDDDDD",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            # Embedded fonts (camera-ready submission requirement)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Save defaults
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


# ---------------------------------------------------------------------------
# Common axis helpers
# ---------------------------------------------------------------------------
def light_grid(ax, axis: str = "y") -> None:
    """Apply consistent, low-contrast grid style."""
    ax.grid(axis=axis, lw=0.4, color="#DDDDDD", linestyle=":", zorder=0)
    ax.set_axisbelow(True)


def hide_top_right(ax) -> None:
    """Hide top + right spines to match the rest of the paper."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def errorbar_style(**overrides):
    """Return matplotlib errorbar kwargs — uniform across figures."""
    base = dict(fmt="none", ecolor=GRAY_DARK, elinewidth=0.9, capsize=2.2)
    base.update(overrides)
    return base


def random_baseline_band(ax, x_lo: float, x_hi: float, *, color: str = GRAY_BG):
    """Shade [x_lo, x_hi] as a random-direction baseline band on the x-axis."""
    ax.axvspan(x_lo, x_hi, color=color, alpha=0.55, zorder=0)
