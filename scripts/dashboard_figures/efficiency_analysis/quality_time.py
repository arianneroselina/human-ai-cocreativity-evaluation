"""Combined quality-time efficiency profiles by practice workflow."""

from __future__ import annotations

from matplotlib import pyplot as plt
import pandas as pd

from scripts.config import (
    QUALITY_Y_MAX,
    QUALITY_Y_MIN,
)
from scripts.dashboard_figures.efficiency_analysis.common import (
    _workflow_efficiency_summary,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import WORKFLOW_COLORS, apply_standard_axes_style, SUBTITLE_FONT_SIZE, \
    VALUE_LABEL_FONT_SIZE
from scripts.utils import save_figure, save_table


def plot_quality_time_efficiency_profile_practice_rounds(
    practice_df: pd.DataFrame,
    time_source: str,
) -> None:
    """Show workflow mean quality against mean total completion time."""
    slug = "17_quality_time_efficiency_profile_practice_rounds"
    summary = _workflow_efficiency_summary(practice_df)
    if summary.empty:
        return

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    annotation_positions = {
        # workflow: (x offset, y offset, horizontal alignment, vertical alignment)
        "human": (-18, -14, "right", "top"),
        "ai": (-18, 12, "right", "bottom"),
        "human_ai": (14, -12, "left", "top"),
        "ai_human": (14, 12, "left", "bottom"),
    }

    for _, row in summary.iterrows():
        workflow = row["workflow"]
        mean_time = float(row["meanCompletionTimeMinutes"])
        mean_quality = float(row["meanQuality"])

        x_low = row["completionTimeCiLow"]
        x_high = row["completionTimeCiHigh"]
        y_low = row["qualityCiLow"]
        y_high = row["qualityCiHigh"]

        xerr = None
        if pd.notna(x_low) and pd.notna(x_high):
            xerr = [[mean_time - x_low], [x_high - mean_time]]

        yerr = None
        if pd.notna(y_low) and pd.notna(y_high):
            yerr = [[mean_quality - y_low], [y_high - mean_quality]]

        ax.errorbar(
            mean_time,
            mean_quality,
            xerr=xerr,
            yerr=yerr,
            fmt="D",
            markersize=10,
            color=WORKFLOW_COLORS[workflow],
            markeredgecolor="black",
            markeredgewidth=1.0,
            capsize=4,
            linewidth=1.2,
            zorder=3,
        )

        x_offset, y_offset, ha, va = annotation_positions.get(
            workflow,
            (10, 10, "left", "bottom"),
        )

        ax.annotate(
            (
                f"{workflow_display_name(workflow)}\n"
                f"{mean_time:.2f} min · {mean_quality:.2f}/5"
            ),
            xy=(mean_time, mean_quality),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=VALUE_LABEL_FONT_SIZE,
            ha=ha,
            va=va,
        )

    ax.annotate(
        "Preferred direction:\nhigher quality, less time",
        xy=(0.05, 0.92),
        xytext=(0.23, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.1,
            "color": "0.35",
        },
        ha="center",
        fontsize=SUBTITLE_FONT_SIZE,
        color="0.30",
    )

    ax.set_title("Quality-Time Efficiency Profile in Practice Rounds")
    ax.set_xlabel("Mean total completion time including pauses (minutes)")
    ax.set_ylabel("Mean overall quality (1-5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.set_xlim(left=0)
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Quality-Time Efficiency Profile in Practice Rounds",
        (
            "Diamonds show workflow means; horizontal and vertical error bars show "
            "descriptive 95% confidence intervals for total completion time and "
            "quality. Workflows nearer the upper-left combine higher quality with "
            f"shorter completion time."
        ),
    )
