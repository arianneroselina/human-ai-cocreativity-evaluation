"""Shared plotting style for the research dashboard."""

import matplotlib

from scripts.config import WORKFLOW_COLORS, WORKFLOW_ORDER


FIGURE_DPI = 220
DEFAULT_FONT_SIZE = 10
TITLE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9
TICK_LABEL_FONT_SIZE = 9

GRID_ALPHA = 0.30
BAR_EDGE_COLOR = "white"

WORKFLOW_PLOT_COLORS = [WORKFLOW_COLORS[workflow] for workflow in WORKFLOW_ORDER]


def configure_matplotlib() -> None:
    """Apply the dashboard-wide Matplotlib defaults once per generation run."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": DEFAULT_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
        }
    )


def apply_standard_axes_style(ax, grid_axis: str = "y") -> None:
    """Apply consistent axes styling to a single plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
