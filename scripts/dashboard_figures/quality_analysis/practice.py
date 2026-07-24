"""Practice-round workflow quality comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    QUALITY_DIMENSION_LABELS,
    QUALITY_PRIMARY_METRIC,
    QUALITY_Y_MAX,
    QUALITY_Y_MIN,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    quality_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import (
    save_figure,
    save_table,
)

from scripts.dashboard_figures.quality_analysis.common import (
    _paired_matrix,
    _plot_two_condition_paired_comparison,
    _plot_workflow_quality_distribution,
    _workflow_order_present,
)


def plot_overall_quality_by_workflow_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare quality across randomized workflows in the practice phase."""
    _plot_workflow_quality_distribution(
        practice_df,
        slug="11_overall_quality_by_workflow_practice_rounds",
        title="Overall Quality by Workflow in Practice Rounds",
        description=(
            "Overall-quality ratings in the randomized practice phase; "
            "points show poems, boxes show distributions, and diamonds show "
            "means with 95% confidence intervals."
        ),
    )


def plot_rating_dimensions_by_workflow_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare evaluator-rated quality dimensions across practice workflows."""
    slug = "12_rating_dimensions_by_workflow_practice_rounds"

    available_dimensions = [
        column
        for column in QUALITY_DIMENSION_LABELS
        if column in practice_df.columns
        and pd.to_numeric(practice_df[column], errors="coerce").notna().any()
    ]
    if not available_dimensions:
        print(
            "Skipping quality dimensions; no configured dimension columns "
            "are available."
        )
        return

    dimension_source = practice_df[["workflow", *available_dimensions]].copy()
    dimension_source[available_dimensions] = dimension_source[
        available_dimensions
    ].apply(pd.to_numeric, errors="coerce")

    long_df = (
        dimension_source.melt(
            id_vars="workflow",
            value_vars=available_dimensions,
            var_name="dimensionColumn",
            value_name="score",
        )
        .dropna(subset=["workflow", "score"])
        .loc[lambda frame: frame["workflow"].isin(WORKFLOW_ORDER)]
    )
    if long_df.empty:
        return

    long_df["dimension"] = long_df["dimensionColumn"].map(QUALITY_DIMENSION_LABELS)

    summary = quality_summary(
        long_df.rename(columns={"score": QUALITY_PRIMARY_METRIC}),
        ["workflow", "dimension"],
    )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    workflows = _workflow_order_present(long_df)
    dimension_labels = [
        QUALITY_DIMENSION_LABELS[column] for column in available_dimensions
    ]
    base_positions = np.arange(len(dimension_labels))
    offsets = (
        np.linspace(-0.26, 0.26, len(workflows))
        if len(workflows) > 1
        else np.array([0.0])
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    for offset, workflow in zip(offsets, workflows):
        workflow_summary = (
            summary.loc[summary["workflow"].eq(workflow)]
            .set_index("dimension")
            .reindex(dimension_labels)
        )
        means = workflow_summary["mean"].to_numpy(dtype=float)
        lows = workflow_summary["ciLow"].to_numpy(dtype=float)
        highs = workflow_summary["ciHigh"].to_numpy(dtype=float)

        ax.errorbar(
            means,
            base_positions + offset,
            xerr=np.vstack(
                [
                    np.where(np.isfinite(lows), means - lows, 0.0),
                    np.where(np.isfinite(highs), highs - means, 0.0),
                ]
            ),
            fmt="o",
            color=WORKFLOW_COLORS[workflow],
            capsize=3,
            markersize=6,
            label=workflow_display_name(workflow),
        )

    ax.set_yticks(base_positions)
    ax.set_yticklabels(dimension_labels)
    ax.invert_yaxis()
    ax.set_xlim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.set_xlabel("Mean evaluator rating (1-5)")
    ax.set_ylabel("Quality dimension")
    ax.set_title("Quality Dimensions by Workflow in Practice Rounds")
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="x")

    save_figure(
        fig,
        slug,
        "Quality Dimensions by Workflow in Practice Rounds",
        (
            "Mean evaluator ratings by workflow and quality dimension in the "
            "randomized practice phase. Error bars show descriptive 95% confidence "
            "intervals."
        ),
    )


def plot_mixed_workflow_direction_quality_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare Human→AI and AI→Human quality within participants."""
    slug = "13_mixed_workflow_direction_quality_practice_rounds"
    paired_df = _paired_matrix(
        practice_df,
        ["human_ai", "ai_human"],
    )

    _plot_two_condition_paired_comparison(
        paired_df,
        left_column="human_ai",
        right_column="ai_human",
        left_label=workflow_display_name("human_ai"),
        right_label=workflow_display_name("ai_human"),
        left_color=WORKFLOW_COLORS["human_ai"],
        right_color=WORKFLOW_COLORS["ai_human"],
        slug=slug,
        title="Mixed-Workflow Direction and Quality in Practice Rounds",
        description=(
            "Within-participant comparison of the two mixed workflows in the "
            "randomized practice phase. Each line connects the same participant."
        ),
    )


def plot_mixed_vs_solo_quality_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare average mixed- and solo-workflow quality within participants."""
    slug = "14_mixed_vs_solo_quality_practice_rounds"

    required_workflows = ["human", "ai", "human_ai", "ai_human"]
    matrix = _paired_matrix(practice_df, required_workflows)
    if matrix.empty:
        return

    paired_df = pd.DataFrame(
        {
            "solo": matrix[["human", "ai"]].mean(axis=1),
            "mixed": matrix[["human_ai", "ai_human"]].mean(axis=1),
        },
        index=matrix.index,
    )

    _plot_two_condition_paired_comparison(
        paired_df,
        left_column="solo",
        right_column="mixed",
        left_label=(
            "Solo workflows\n"
            f"({workflow_display_name('human')} + "
            f"{workflow_display_name('ai')})"
        ),
        right_label=(
            "Mixed workflows\n"
            f"({workflow_display_name('human_ai')} + "
            f"{workflow_display_name('ai_human')})"
        ),
        left_color="0.45",
        right_color=WORKFLOW_COLORS["human_ai"],
        slug=slug,
        title="Mixed versus Solo Workflow Quality in Practice Rounds",
        description=(
            "Within-participant comparison of average quality across the two solo "
            "and two mixed workflows in the randomized practice phase."
        ),
    )
