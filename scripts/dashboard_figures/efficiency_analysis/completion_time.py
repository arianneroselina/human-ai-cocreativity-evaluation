"""Total completion-time distributions by practice workflow."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.dashboard_figures.efficiency_analysis.common import (
    _workflow_efficiency_summary,
    _workflow_order_present,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import save_figure, save_table


def plot_completion_time_by_workflow_practice_rounds(
    practice_df: pd.DataFrame,
    time_source: str,
) -> None:
    """Show total completion-time distributions for each practice workflow."""
    slug = "16_completion_time_by_workflow_practice_rounds"
    workflows = _workflow_order_present(practice_df)
    if not workflows:
        return

    summary = _workflow_efficiency_summary(practice_df)
    save_table(summary, slug, index=False)

    box_data = [
        practice_df.loc[
            practice_df["workflow"].eq(workflow),
            "totalCompletionTimeMinutes",
        ]
        .dropna()
        .to_numpy()
        for workflow in workflows
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    boxplot = ax.boxplot(
        box_data,
        tick_labels=[workflow_display_name(workflow) for workflow in workflows],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={"marker": "", "markersize": 0},
    )

    for patch, workflow in zip(boxplot["boxes"], workflows):
        patch.set_facecolor(WORKFLOW_COLORS[workflow])
        patch.set_alpha(0.35)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    rng = np.random.default_rng(42)
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        values = (
            practice_df.loc[
                practice_df["workflow"].eq(workflow),
                "totalCompletionTimeMinutes",
            ]
            .dropna()
            .to_numpy()
        )
        jitter = rng.uniform(-0.13, 0.13, size=len(values))

        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=WORKFLOW_COLORS[workflow],
            alpha=0.48,
            s=28,
            linewidths=0,
            zorder=3,
        )

        row = indexed_summary.loc[workflow]
        mean = float(row["meanCompletionTimeMinutes"])
        low = row["completionTimeCiLow"]
        high = row["completionTimeCiHigh"]

        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color="black",
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=4,
                zorder=5,
            )
        else:
            ax.scatter(
                position,
                mean,
                marker="D",
                color="black",
                facecolor="white",
                zorder=5,
            )

        ax.text(
            position,
            0.02,
            f"n={int(row['count'])}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    mean_handle = Line2D(
        [],
        [],
        marker="D",
        color="black",
        markerfacecolor="white",
        linestyle="None",
        label="Mean ± 95% CI",
    )
    ax.legend(handles=[mean_handle], loc="upper right")
    ax.set_title("Total Completion Time by Workflow in Practice Rounds")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Total completion time including pauses (minutes)")
    ax.set_ylim(bottom=0)
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Total Completion Time by Workflow in Practice Rounds",
        (
            "Practice round completion times include pauses. Points show outputs, "
            "boxes show distributions, and diamonds show descriptive means with "
            f"95% confidence intervals."
        ),
    )
