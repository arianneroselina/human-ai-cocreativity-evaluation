"""Complete constraint-pass rate by practice workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.config import WORKFLOW_ORDER
from scripts.dashboard_figures.helpers import (
    pass_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.style import WORKFLOW_COLORS, VALUE_LABEL_FONT_SIZE
from scripts.utils import save_figure, save_table


def plot_practice_constraint_pass_rate_by_workflow(practice_df) -> None:
    """Compare complete constraint-fulfilment rates across practice workflows.

    Only non-empty submitted poems are included. Each point is the observed
    proportion of submitted poems in which every applicable task constraint
    was fulfilled. Horizontal whiskers show 95% Wilson confidence intervals.
    """
    slug = "21_practice_constraint_pass_rate_by_workflow"

    # Constraint fulfilment is evaluated only for non-empty submitted poems.
    evaluated = practice_df[
        practice_df["wordCount"].fillna(0).gt(0)
    ].copy()

    if evaluated.empty:
        return

    summary = pass_summary(evaluated, ["workflow"])
    if summary.empty:
        return

    summary = (
        summary.set_index("workflow")
        .reindex(WORKFLOW_ORDER)
        .dropna(subset=["totalRounds"])
        .reset_index()
    )

    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)

    # Rename output columns so the saved table also reflects that
    # the denominator is submitted poems rather than all rounds.
    summary = summary.rename(
        columns={
            "passedRounds": "passedPoems",
            "totalRounds": "totalPoems",
        }
    )

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    y_positions = np.arange(len(summary))
    values = summary["passRatePercent"].to_numpy(dtype=float)
    lower_ci = summary["lowerCI"].to_numpy(dtype=float)
    upper_ci = summary["upperCI"].to_numpy(dtype=float)

    lower_errors = values - lower_ci
    upper_errors = upper_ci - values

    # Light reference lines make percentage comparisons easier.
    for x_position in [0, 25, 50, 75, 100]:
        ax.axvline(
            x_position,
            color="#e6e6e6",
            linewidth=0.9,
            zorder=0,
        )

    # Confidence intervals first, so points remain clear above them.
    ax.errorbar(
        values,
        y_positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        zorder=2,
    )

    # One observed pass-rate point per workflow.
    for position, (_, row) in enumerate(summary.iterrows()):
        workflow_color = WORKFLOW_COLORS[row["workflow"]]

        ax.scatter(
            row["passRatePercent"],
            position,
            s=95,
            color=workflow_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        label = (
            f"{int(row['passedPoems'])}/{int(row['totalPoems'])} "
            f"({row['passRatePercent']:.1f}%)"
        )

        ax.annotate(
            label,
            (row["upperCI"], position),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=VALUE_LABEL_FONT_SIZE,
            color="#333333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["workflowLabel"])
    ax.invert_yaxis()

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_xlabel("Submitted poems fully meeting every constraint", labelpad=10)

    ax.set_title(
        "Complete Constraint Fulfilment Rate by Workflow in Practice Rounds"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=10,
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        top=0.84,
        bottom=0.18,
    )

    save_figure(
        fig,
        slug,
        "Complete Constraint Fulfilment Rate by Workflow in Practice Rounds",
        "Points show the observed percentage of non-empty submitted practice-round "
        "poems that fulfilled every applicable task constraint. Horizontal whiskers "
        "show 95% Wilson confidence intervals. Labels show the number of fully "
        "successful poems out of all non-empty submitted poems for each workflow.",
    )
