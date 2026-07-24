"""Final stated workflow-ranking figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    WORKFLOW_COLORS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    ranking_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    RANK_COLORS,
    apply_standard_axes_style,
)
from scripts.utils import save_figure, save_table


def plot_final_workflow_preference(
    ranking_rows: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    """Visualise average stated rank and the full rank distribution."""
    slug = "08_final_workflow_preference"

    if not audit_df.empty:
        save_table(audit_df, f"{slug}_ranking_audit", index=False)

    if ranking_rows.empty:
        return

    summary = ranking_summary(ranking_rows)
    rank_columns = list(range(1, len(WORKFLOW_ORDER) + 1))

    # Lower mean rank indicates stronger preference.
    workflow_order = summary["meanRank"].sort_values().index.tolist()
    display_summary = summary.loc[workflow_order].copy()

    valid_participants = ranking_rows["sessionId"].nunique()

    # Export counts and mean ranks.
    export_df = display_summary.rename(index=WORKFLOW_LABELS).rename(
        columns={rank: f"Rank {rank}" for rank in rank_columns}
    )
    save_table(export_df, slug)

    # Convert rank counts to percentages for the distribution panel.
    rank_counts = display_summary[rank_columns]
    row_totals = rank_counts.sum(axis=1)
    rank_percentages = rank_counts.div(row_totals, axis=0) * 100

    positions = np.arange(len(workflow_order))

    fig, (ax_mean, ax_distribution) = plt.subplots(
        ncols=2,
        figsize=(11.8, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.8]},
    )

    # ------------------------------------------------------------
    # Left panel: average rank
    # ------------------------------------------------------------
    mean_ranks = display_summary["meanRank"].to_numpy()

    ax_mean.hlines(
        y=positions,
        xmin=1,
        xmax=mean_ranks,
        color="0.75",
        linewidth=2,
        zorder=1,
    )

    for index, workflow in enumerate(workflow_order):
        mean_rank = display_summary.loc[workflow, "meanRank"]
        first_choice_count = int(display_summary.loc[workflow, 1])

        ax_mean.scatter(
            mean_rank,
            index,
            s=95,
            color=WORKFLOW_COLORS[workflow],
            edgecolor=BAR_EDGE_COLOR,
            zorder=2,
        )

        ax_mean.text(
            mean_rank + 0.08,
            index,
            f"{mean_rank:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        ax_mean.text(
            mean_rank + 0.08,
            index + 0.16,
            f"{first_choice_count} first-choice votes",
            va="center",
            fontsize=8,
            color="0.35",
        )

    ax_mean.set_yticks(positions)
    ax_mean.set_yticklabels(
        [
            f"{position + 1}. {workflow_display_name(workflow)}"
            for position, workflow in enumerate(workflow_order)
        ]
    )
    ax_mean.invert_yaxis()

    ax_mean.set_xlim(0.8, 4.35)
    ax_mean.set_xticks([1, 2, 3, 4])
    ax_mean.set_xticklabels(["1\nBest", "2", "3", "4\nWorst"])
    ax_mean.set_xlabel("Average assigned rank")
    ax_mean.set_title("Average preference rank")

    apply_standard_axes_style(ax_mean, grid_axis="x")

    # ------------------------------------------------------------
    # Right panel: complete rank distribution
    # ------------------------------------------------------------
    left = np.zeros(len(workflow_order))

    for rank in rank_columns:
        percentages = rank_percentages[rank].to_numpy()

        bars = ax_distribution.barh(
            positions,
            percentages,
            left=left,
            label=f"Rank {rank}",
            color=RANK_COLORS[rank - 1],
            edgecolor=BAR_EDGE_COLOR,
        )

        # Add percentages only where the segment is wide enough.
        for index, (bar, percentage) in enumerate(zip(bars, percentages)):
            if percentage >= 8:
                ax_distribution.text(
                    left[index] + percentage / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        left += percentages

    ax_distribution.set_xlim(0, 100)
    ax_distribution.set_xlabel("Participants assigning each rank (%)")
    ax_distribution.set_title("Distribution of assigned ranks")

    apply_standard_axes_style(ax_distribution, grid_axis="x")

    ax_distribution.legend(
        title="Assigned rank",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.suptitle(
        f"Final Workflow Preference (N={valid_participants})",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(
        fig,
        slug,
        "Final Workflow Preference",
        (
            f"Average and distribution of final workflow rankings from "
            f"{valid_participants} participants. Rank 1 represents the strongest "
            "preference. Each participant ranked every workflow once; rankings "
            "were not weighted by the frequency of workflow use in the main rounds."
        ),
    )
