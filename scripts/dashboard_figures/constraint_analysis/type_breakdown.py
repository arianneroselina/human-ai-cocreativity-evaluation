"""Constraint failure rates by requirement type and workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import WORKFLOW_ORDER
from scripts.dashboard_figures.constraint_analysis.common import (
    CONSTRAINT_TYPE_ORDER,
    _explode_requirement_results,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import VALUE_LABEL_FONT_SIZE
from scripts.utils import save_figure, save_table


def plot_practice_failure_breakdown_by_constraint_type(practice_df) -> None:
    """Identify requirement types that produce failures under each workflow."""
    slug = "23_practice_failure_breakdown_by_constraint_type"

    exploded = _explode_requirement_results(practice_df)
    if exploded.empty:
        return

    summary = (
        exploded.groupby(["workflow", "constraintType"])["passed"]
        .agg(totalChecks="count", passedChecks="sum")
        .reset_index()
    )
    summary["failureRatePercent"] = (
        1 - summary["passedChecks"] / summary["totalChecks"]
    ) * 100
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    observed_types = [
        constraint_type
        for constraint_type in CONSTRAINT_TYPE_ORDER
        if constraint_type in set(summary["constraintType"])
    ]
    workflow_order = [
        workflow for workflow in WORKFLOW_ORDER if workflow in set(summary["workflow"])
    ]

    failure_matrix = summary.pivot(
        index="workflow",
        columns="constraintType",
        values="failureRatePercent",
    ).reindex(index=workflow_order, columns=observed_types)
    check_matrix = summary.pivot(
        index="workflow",
        columns="constraintType",
        values="totalChecks",
    ).reindex(index=workflow_order, columns=observed_types)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.55 * len(observed_types) + 2.8), 4.7))
    masked = np.ma.masked_invalid(failure_matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, vmin=0, vmax=100, cmap="Reds", aspect="auto")

    for row_index, workflow in enumerate(workflow_order):
        for col_index, constraint_type in enumerate(observed_types):
            rate = failure_matrix.iloc[row_index, col_index]
            checks = check_matrix.iloc[row_index, col_index]
            label = "–" if pd.isna(rate) else f"{rate:.0f}%\nn={int(checks)}"
            text_color = "white" if not pd.isna(rate) and rate >= 55 else "black"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=VALUE_LABEL_FONT_SIZE,
                color=text_color,
            )

    ax.set_xticks(range(len(observed_types)))
    ax.set_xticklabels(observed_types, rotation=20, ha="right")
    ax.set_yticks(range(len(workflow_order)))
    ax.set_yticklabels([workflow_display_name(workflow) for workflow in workflow_order])
    ax.set_xlabel("Constraint type")
    ax.set_ylabel("Assigned workflow")
    ax.set_title("Constraint Failure Breakdown by Type in Practice Rounds")
    fig.colorbar(image, ax=ax, label="Failure rate (%)")

    save_figure(
        fig,
        slug,
        "Constraint Failure Breakdown by Type in Practice Rounds",
        "Failure rates by requirement type and assigned workflow. Each cell also "
        "shows the number of individual requirement checks supporting the rate.",
    )
