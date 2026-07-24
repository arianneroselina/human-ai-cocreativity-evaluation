"""Subjective experience during the randomized practice rounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    AI_EXPERIENCE_METRICS,
    SATISFACTION_COLUMN,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.series import plot_workflow_round_series
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.dashboard_figures.summaries import grouped_metric_summary
from scripts.utils import require_columns, save_figure, save_table

from scripts.dashboard_figures.experience_analysis.common import _available_rounds


def plot_satisfaction_by_practice_round_and_workflow(
    practice_df: pd.DataFrame,
) -> None:
    """Show satisfaction across randomized practice round positions."""
    slug = "31_satisfaction_by_practice_round_and_workflow"

    required_columns = {SATISFACTION_COLUMN, "roundIndex", "workflow"}
    if not require_columns(
        practice_df,
        required_columns,
        "practice round satisfaction by workflow",
    ):
        return

    plot_df = practice_df.dropna(
        subset=[SATISFACTION_COLUMN, "roundIndex", "workflow"]
    ).copy()
    if plot_df.empty:
        return

    rounds = _available_rounds(plot_df)
    summary = grouped_metric_summary(
        plot_df,
        group_columns=["roundIndex", "workflow"],
        metric_columns=[SATISFACTION_COLUMN],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.6, 5.3))

    for workflow in WORKFLOW_ORDER:
        plot_workflow_round_series(
            ax,
            summary,
            workflow,
            rounds,
            SATISFACTION_COLUMN,
        )

    ax.set_title("Participant Satisfaction by Practice Round and Workflow")
    ax.set_xlabel("Practice round")
    ax.set_ylabel("Satisfaction rating (1-5)")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Practice {index + 1}" for index in range(len(rounds))])
    ax.set_ylim(0.7, 5.42)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(
        title="Assigned workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    apply_standard_axes_style(ax, grid_axis="y")
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    save_figure(
        fig,
        slug,
        "Participant Satisfaction by Practice Round and Workflow",
        (
            "Mean satisfaction across the randomized practice phase, grouped by "
            "practice round and assigned workflow. Error bars show "
            "approximate 95% confidence intervals; labels show observation counts."
        ),
    )


def plot_ai_experience_by_practice_round_and_workflow(
    practice_df: pd.DataFrame,
) -> None:
    """Show AI interaction ratings across randomized practice rounds."""
    slug = "32_ai_experience_by_practice_round_and_workflow"

    available_metrics = [
        metric
        for metric in AI_EXPERIENCE_METRICS
        if metric in practice_df.columns and practice_df[metric].notna().any()
    ]
    if not available_metrics:
        return

    ai_workflows = [workflow for workflow in WORKFLOW_ORDER if workflow != "human"]
    plot_df = practice_df.loc[practice_df["workflow"].isin(ai_workflows)].copy()
    if plot_df.empty:
        return

    rounds = _available_rounds(plot_df)
    summary = grouped_metric_summary(
        plot_df,
        group_columns=["roundIndex", "workflow"],
        metric_columns=available_metrics,
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["metricLabel"] = summary["metric"].map(AI_EXPERIENCE_METRICS)
    save_table(summary, slug, index=False)

    metric_count = len(available_metrics)
    n_columns = 2
    n_rows = int(np.ceil(metric_count / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.4, 4.45 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, metric in zip(axes_flat, available_metrics):
        for workflow in ai_workflows:
            plot_workflow_round_series(
                axis,
                summary,
                workflow,
                rounds,
                metric,
            )

        axis.set_title(AI_EXPERIENCE_METRICS[metric])
        axis.set_xticks(rounds)
        axis.set_xticklabels([f"Practice {index + 1}" for index in range(len(rounds))])
        axis.set_ylim(0.7, 5.42)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Practice round")
        axis.set_ylabel("Participant rating (1-5)")
        apply_standard_axes_style(axis, grid_axis="y")

    for axis in axes_flat[len(available_metrics) :]:
        axis.set_visible(False)

    legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        title="Assigned AI-supported workflow",
        bbox_to_anchor=(0.99, 0.5),
        loc="center left",
    )
    fig.suptitle(
        "AI Interaction Ratings by Practice Round and Workflow",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 0.82, 0.97))

    save_figure(
        fig,
        slug,
        "AI Interaction Ratings by Practice Round and Workflow",
        (
            "Mean ratings for AI understanding, collaboration quality, creativity "
            "support, and overall AI performance across the randomized practice "
            "phase. Error bars show approximate 95% confidence intervals; labels "
            "show observation counts."
        ),
    )


def plot_tlx_score_by_workflow_in_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare Raw NASA-TLX workload across practice workflows.

    Raw NASA-TLX is the equally weighted mean of the six NASA-TLX subscales
    for each participant-round. Scores remain on the original 0-20 scale.
    """
    slug = "33_tlx_score_by_workflow_in_practice_rounds"

    required_columns = {"workflow", "rawNasaTlxScore"}
    if not require_columns(
        practice_df,
        required_columns,
        "Raw NASA-TLX score by practice workflow",
    ):
        return

    plot_df = practice_df.dropna(subset=["workflow", "rawNasaTlxScore"]).copy()
    if plot_df.empty:
        return

    summary = grouped_metric_summary(
        plot_df,
        group_columns=["workflow"],
        metric_columns=["rawNasaTlxScore"],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    workflow_order = [
        workflow for workflow in WORKFLOW_ORDER if workflow in set(summary["workflow"])
    ]
    plot_summary = (
        summary.set_index("workflow")
        .reindex(workflow_order)
        .dropna(subset=["mean"])
        .reset_index()
    )
    if plot_summary.empty:
        return

    positions = np.arange(len(plot_summary))
    values = plot_summary["mean"].to_numpy(dtype=float)
    lower_ci = plot_summary["lowerCI"].to_numpy(dtype=float)
    upper_ci = plot_summary["upperCI"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.7, 4.8))

    ax.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lower_ci, upper_ci - values]),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        zorder=2,
    )

    for position, (_, row) in zip(positions, plot_summary.iterrows()):
        workflow = row["workflow"]

        ax.scatter(
            row["mean"],
            position,
            s=95,
            color=WORKFLOW_COLORS[workflow],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{row['mean']:.1f} (n={int(row['count'])})",
            (row["mean"], position),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [workflow_display_name(workflow) for workflow in plot_summary["workflow"]]
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 21.8)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Raw NASA-TLX workload score (0-20)")
    ax.set_ylabel("Assigned workflow")
    ax.set_title("Raw NASA-TLX Workload by Workflow in Practice Rounds")
    apply_standard_axes_style(ax, grid_axis="x")

    save_figure(
        fig,
        slug,
        "Raw NASA-TLX Workload by Workflow in Practice Rounds",
        (
            "Raw NASA-TLX is the equally weighted mean of the six subscales for "
            "each practice round observation. Higher scores indicate greater "
            "perceived workload. Points show workflow means and whiskers show "
            "approximate 95% confidence intervals."
        ),
    )
