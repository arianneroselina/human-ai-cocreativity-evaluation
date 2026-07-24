"""First Main-round workflow choice and error-exposure opportunity."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    FOOTNOTE_TEXT_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    save_figure,
    save_table,
)


def plot_main_round1_workflow_choice(prepared) -> None:
    """Show voluntary workflow choices in the first Main round."""
    slug = "101_main_round1_workflow_choice"

    error_round = prepared[prepared["roundIndex"].eq(INJECTED_ERROR_ROUND_INDEX)].copy()
    if error_round.empty:
        return

    summary = (
        error_round.groupby("workflow")
        .size()
        .reindex(WORKFLOW_ORDER, fill_value=0)
        .rename("participantCount")
        .reset_index()
    )

    summary = summary[summary["participantCount"].gt(0)].copy()

    total_participants = int(summary["participantCount"].sum())
    if total_participants == 0:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["percentage"] = summary["participantCount"] / total_participants * 100

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    bars = ax.bar(
        summary["workflowLabel"],
        summary["participantCount"],
        color=[WORKFLOW_COLORS[workflow] for workflow in summary["workflow"]],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{int(row['participantCount'])}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, summary["participantCount"].max() + 5)
    ax.set_xlabel("Workflow selected")
    ax.set_ylabel("Participants")
    ax.set_title("Workflow Choices in Main Round 1")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    apply_standard_axes_style(ax, grid_axis="y")

    fig.text(
        0.01,
        0.01,
        (
            f"n = {total_participants}. AI-supported selections in this round "
            "were the opportunity for participants to encounter the injected error."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=FOOTNOTE_TEXT_COLOR,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Workflow Choices in Main Round 1",
        "Voluntary workflow selections in the first Main round. Participants "
        "selecting an AI-supported workflow encountered the injected-error condition.",
    )
