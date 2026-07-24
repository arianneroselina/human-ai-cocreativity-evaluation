"""Complete constraint-pass rate by practice workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scripts.config import WORKFLOW_COLORS, WORKFLOW_ORDER
from scripts.dashboard_figures.helpers import (
    pass_summary,
    workflow_display_name,
)
from scripts.utils import save_figure, save_table


def plot_practice_constraint_pass_rate_by_workflow(practice_df) -> None:
    """Compare complete constraint-fulfillment rates across practice workflows.

    Each point is the observed proportion of practice rounds in which every
    task constraint was fulfilled. Horizontal whiskers show 95% Wilson
    confidence intervals.
    """
    slug = "21_practice_constraint_pass_rate_by_workflow"

    summary = pass_summary(practice_df, ["workflow"])
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
            f"{int(row['passedRounds'])}/{int(row['totalRounds'])} "
            f"({row['passRatePercent']:.1f}%)"
        )

        ax.text(
            105,
            position,
            label,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["workflowLabel"], fontsize=10)
    ax.invert_yaxis()

    # Here intermediate values are valid because this axis represents
    # workflow-level pass-rate estimates, not the individual binary outcomes.
    ax.set_xlim(-2, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_xlabel("Rounds fully meeting every constraint", labelpad=10)

    ax.set_title("Complete Constraint Fulfillment Rate by Workflow in Practice Rounds")

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
        "Complete Constraint Fulfillment Rate by Workflow in Practice Rounds",
        "Points show the observed percentage of practice-round poems that fulfilled "
        "every task constraint. Horizontal whiskers show 95% Wilson confidence "
        "intervals. Labels show the number of fully successful rounds out of all "
        "evaluated rounds for each workflow.",
    )
