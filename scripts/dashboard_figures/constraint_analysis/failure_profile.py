"""Mutually exclusive constraint-failure profiles by workflow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import WORKFLOW_ORDER
from scripts.dashboard_figures.constraint_analysis.common import (
    FAILURE_PROFILE_COLORS,
    FAILURE_PROFILE_ORDER,
    _constraint_failure_profile,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import apply_standard_axes_style, VALUE_LABEL_FONT_SIZE
from scripts.utils import save_figure, save_table


def plot_practice_constraint_failure_profile_by_workflow(practice_df) -> None:
    """Show why practice-round outputs did not fully meet all constraints."""
    slug = "22_practice_constraint_failure_profile_by_workflow"

    profile_df = practice_df.dropna(subset=["passedNumeric", "workflow"]).copy()

    profile_df["failureProfile"] = [
        _constraint_failure_profile(
            row.passedNumeric,
            row.requirementResults,
        )
        for row in profile_df.itertuples(index=False)
    ]

    profile_df = profile_df.dropna(subset=["failureProfile"]).copy()
    if profile_df.empty:
        return

    workflow_order = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow in set(profile_df["workflow"])
    ]

    observed_profiles = [
        profile
        for profile in FAILURE_PROFILE_ORDER
        if profile in set(profile_df["failureProfile"])
    ]

    if not workflow_order or not observed_profiles:
        return

    counts = pd.crosstab(
        profile_df["workflow"],
        profile_df["failureProfile"],
    ).reindex(
        index=workflow_order,
        columns=observed_profiles,
        fill_value=0,
    )

    counts = counts.loc[counts.sum(axis=1).gt(0)]
    if counts.empty:
        return

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    summary = (
        counts.rename_axis(
            index="workflow",
            columns="failureProfile",
        )
        .stack()
        .rename("rounds")
        .reset_index()
    )
    summary["totalRounds"] = summary["workflow"].map(counts.sum(axis=1))
    summary["percentage"] = summary["rounds"] / summary["totalRounds"] * 100
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    positions = np.arange(len(counts))
    left = np.zeros(len(counts), dtype=float)

    for profile in observed_profiles:
        values = percentages[profile].to_numpy(dtype=float)
        round_counts = counts[profile].to_numpy(dtype=int)

        ax.barh(
            positions,
            values,
            left=left,
            color=FAILURE_PROFILE_COLORS[profile],
            edgecolor="white",
            linewidth=0.8,
            label=profile,
        )

        for position, value, count, start in zip(
            positions,
            values,
            round_counts,
            left,
        ):
            if value >= 9:
                ax.text(
                    start + value / 2,
                    position,
                    f"{count}\n{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=VALUE_LABEL_FONT_SIZE,
                )

        left += values

    ax.set_yticks(positions)
    ax.set_yticklabels([workflow_display_name(workflow) for workflow in counts.index])
    ax.invert_yaxis()

    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of practice-round outputs (%)")
    ax.set_title("Constraint Failure Profiles by Workflow in Practice Rounds")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False
    )

    apply_standard_axes_style(ax, grid_axis="x")

    fig.subplots_adjust(
        left=0.21,
        right=0.95,
        top=0.86,
        bottom=0.28,
    )

    save_figure(
        fig,
        slug,
        "Constraint Failure Profiles by Workflow in Practice Rounds",
        "Each bar represents all practice-round outputs within one assigned workflow. "
        "Categories are mutually exclusive. “Line-count rule only” means that "
        "the line-count check was the sole failed requirement; “Multiple rules "
        "failed” can include the line-count rule alongside other failed checks.",
    )
