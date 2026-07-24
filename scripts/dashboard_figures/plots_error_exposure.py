"""Injected AI-error exposure analysis.

Figures
-------
101  Main Round 1 workflow choice and exposure opportunity
102  Participant satisfaction across Main rounds by exposure and workflow
103  AI interaction ratings across Main rounds by exposure and workflow
104  Raw NASA-TLX workload by workflow and exposure in Main rounds
105  Post-error workflow choices by Main Round 1 exposure
106  Main round quality patterns by AI-error exposure
107  Complete constraint fulfillment across Main rounds by exposure
108  Line-count failure pattern across Main rounds by exposure
109  Final workflow preference by reported AI errors
110  Awareness of the injected error among exposed interview respondents
111  Other AI error types reported in interviews

Exposure is determined by the workflow voluntarily selected in Main Round 1.
All exposure-group comparisons are descriptive rather than randomized causal
effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    ERROR_ROUND_INDEX,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
    AWARENESS_LABELS,
    OTHER_AI_ERROR_LABELS,
    MAIN_ROUND_INDICES,
    QUALITY_Y_MIN,
    QUALITY_Y_MAX,
    SATISFACTION_COLUMN,
    AI_EXPERIENCE_METRICS,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    workflow_display_name,
    round_display_name,
    build_valid_ranking_rows,
    ranking_summary,
    ordered_exposure_groups,
    quality_summary,
    phase_data,
    pass_summary,
    parse_requirement_results,
    wilson_interval,
    add_passed_numeric,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.style import BAR_EDGE_COLOR, apply_standard_axes_style
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
    parse_bool_or_none,
)




# ---------------------------------------------------------------------------
# Shared descriptive summaries
# ---------------------------------------------------------------------------


def _mean_ci_summary(
        dataframe: pd.DataFrame,
        group_columns: list[str],
        metric_columns: list[str],
) -> pd.DataFrame:
    """Return descriptive means and approximate 95% confidence intervals."""
    rows = []

    for group_values, group_df in dataframe.groupby(
            group_columns,
            dropna=False,
            observed=True,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_dict = dict(zip(group_columns, group_values))

        for metric in metric_columns:
            values = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            count = int(len(values))
            mean = float(values.mean())
            standard_deviation = (
                float(values.std(ddof=1)) if count > 1 else np.nan
            )
            standard_error = (
                standard_deviation / np.sqrt(count)
                if count > 1
                else np.nan
            )
            margin = (
                1.96 * standard_error
                if np.isfinite(standard_error)
                else np.nan
            )

            rows.append(
                {
                    **group_dict,
                    "metric": metric,
                    "mean": mean,
                    "standardDeviation": standard_deviation,
                    "count": count,
                    "lowerCI": (
                        mean - margin if np.isfinite(margin) else np.nan
                    ),
                    "upperCI": (
                        mean + margin if np.isfinite(margin) else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def _plot_round_metric_series(
        ax,
        summary: pd.DataFrame,
        *,
        workflow: str,
        rounds: list[int],
        metric: str,
) -> None:
    """Plot one workflow series while preserving missing round-workflow cells."""
    workflow_summary = (
        summary.loc[
            summary["workflow"].eq(workflow)
            & summary["metric"].eq(metric)
            ]
        .set_index("roundIndex")
        .reindex(rounds)
    )

    valid = workflow_summary["mean"].notna().to_numpy()
    if not valid.any():
        return

    x_values = np.asarray(rounds, dtype=float)
    means = workflow_summary["mean"].to_numpy(dtype=float)

    ax.plot(
        x_values,
        means,
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        color=WORKFLOW_COLORS[workflow],
        label=workflow_display_name(workflow),
        zorder=3,
    )

    valid_ci = (
            workflow_summary["lowerCI"].notna()
            & workflow_summary["upperCI"].notna()
    ).to_numpy()

    if valid_ci.any():
        ci_rows = workflow_summary.loc[
            workflow_summary["lowerCI"].notna()
            & workflow_summary["upperCI"].notna()
            ]
        lower = means[valid_ci] - ci_rows["lowerCI"].to_numpy(dtype=float)
        upper = ci_rows["upperCI"].to_numpy(dtype=float) - means[valid_ci]

        ax.errorbar(
            x_values[valid_ci],
            means[valid_ci],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=WORKFLOW_COLORS[workflow],
            capsize=3,
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )


def _main_round_labels(rounds: list[int]) -> list[str]:
    """Return sequential labels for observed Main round indices."""
    return [f"Main {index + 1}" for index in range(len(rounds))]


# ---------------------------------------------------------------------------
# 101: Main Round 1 workflow choice and exposure opportunity
# ---------------------------------------------------------------------------


def plot_main_round1_workflow_choice(prepared) -> None:
    """Show voluntary workflow choices in the first Main round."""
    slug = "101_main_round1_workflow_choice"

    error_round = prepared[prepared["roundIndex"].eq(ERROR_ROUND_INDEX)].copy()
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
        color="#4A4A4A",
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Workflow Choices in Main Round 1",
        "Voluntary workflow selections in the first Main round. Participants "
        "selecting an AI-supported workflow encountered the injected-error condition.",
    )


# ---------------------------------------------------------------------------
# 102: Participant satisfaction across Main rounds
# ---------------------------------------------------------------------------


def plot_main_round_satisfaction_by_exposure(
        main_df: pd.DataFrame,
) -> None:
    """Show Main round satisfaction by exposure group and selected workflow.

    Each point represents a separate round-workflow mean. Points are not
    connected because the participants selecting a workflow can change between
    Main rounds.
    """
    slug = "102_main_round_satisfaction_by_exposure_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        SATISFACTION_COLUMN,
    }
    if not require_columns(
            main_df,
            required,
            "Main round satisfaction by error exposure",
    ):
        return

    plot_df = main_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            SATISFACTION_COLUMN,
        ]
    ).copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )
    plot_df[SATISFACTION_COLUMN] = pd.to_numeric(
        plot_df[SATISFACTION_COLUMN],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=["roundIndex", SATISFACTION_COLUMN]
    )
    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)
    rounds = sorted(plot_df["roundIndex"].unique().tolist())

    if not groups or not rounds:
        return

    summary = _mean_ci_summary(
        plot_df,
        group_columns=[
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
        metric_columns=[SATISFACTION_COLUMN],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )
    save_table(summary, slug, index=False)

    # Slightly separate workflow estimates around each Main round position.
    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(-0.18, 0.18, len(WORKFLOW_ORDER)),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.6 * len(groups), 5.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(group)
            & summary["metric"].eq(SATISFACTION_COLUMN)
            ]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            valid = workflow_summary["mean"].notna().to_numpy()
            if not valid.any():
                continue

            x_values = (
                    np.asarray(rounds, dtype=float)
                    + workflow_offsets[workflow]
            )
            means = workflow_summary["mean"].to_numpy(dtype=float)

            # Separate points: no connecting lines.
            axis.scatter(
                x_values[valid],
                means[valid],
                s=48,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.7,
                label=workflow_display_name(workflow),
                zorder=4,
            )

            ci_valid = (
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna()
            ).to_numpy()

            if ci_valid.any():
                lower_ci = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna(),
                    "lowerCI",
                ].to_numpy(dtype=float)

                upper_ci = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna(),
                    "upperCI",
                ].to_numpy(dtype=float)

                axis.errorbar(
                    x_values[ci_valid],
                    means[ci_valid],
                    yerr=np.vstack(
                        [
                            means[ci_valid] - lower_ci,
                            upper_ci - means[ci_valid],
                            ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.1,
                    capsize=3,
                    alpha=0.9,
                    zorder=3,
                )

            valid_rows = workflow_summary.loc[
                workflow_summary["mean"].notna()
            ]

            for x_value, y_value, count in zip(
                    x_values[valid],
                    means[valid],
                    valid_rows["count"].to_numpy(dtype=int),
            ):
                axis.annotate(
                    f"n={count}",
                    (x_value, y_value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                )

        axis.axvline(
            ERROR_ROUND_INDEX,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.55,
            zorder=1,
        )

        axis.set_title(exposure_display_name(group))
        axis.set_xticks(rounds)
        axis.set_xticklabels(_main_round_labels(rounds))
        axis.set_xlabel("Main round")
        axis.set_ylim(0.7, 5.42)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        apply_standard_axes_style(axis, grid_axis="y")

    axes[0].set_ylabel("Satisfaction rating (1–5)")

    # Collect all workflow labels that appear in either exposure panel.
    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "Participant Satisfaction by Main Round, Workflow, and Error Exposure",
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate round-workflow means and are not connected "
            "because participants could switch workflows between rounds. "
            "The dashed line marks the injected-error round. Cells with very "
            "small sample sizes should be interpreted cautiously."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )

    fig.tight_layout(rect=(0, 0.06, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Participant Satisfaction by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean satisfaction estimates for each Main round workflow "
            "cell, shown by actual error-exposure group. Points are not connected "
            "because workflow-group membership could change between rounds. "
            "Error bars show approximate 95% confidence intervals; cells with one "
            "observation have no confidence interval."
        ),
    )


# ---------------------------------------------------------------------------
# 103: AI interaction ratings across Main rounds
# ---------------------------------------------------------------------------


def plot_main_round_ai_experience_by_exposure(
        main_df: pd.DataFrame,
) -> None:
    """Show AI interaction ratings by Main round, exposure, and workflow.

    Each point represents a separate Main round and workflow subgroup.
    Points are not connected because participants could switch workflows
    between Main rounds.
    """
    slug = "103_main_round_ai_experience_by_exposure_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
    }
    if not require_columns(
            main_df,
            required,
            "AI interaction ratings in Main rounds",
    ):
        return

    available_metrics = [
        metric
        for metric in AI_EXPERIENCE_METRICS
        if metric in main_df.columns
           and pd.to_numeric(
            main_df[metric],
            errors="coerce",
        ).notna().any()
    ]
    if not available_metrics:
        return

    ai_workflows = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow != "human"
    ]

    plot_df = main_df.loc[
        main_df["workflow"].isin(ai_workflows)
    ].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(
            plot_df[metric],
            errors="coerce",
        )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
        ]
    )
    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)
    rounds = sorted(plot_df["roundIndex"].unique().tolist())

    if not groups or not rounds:
        return

    summary = _mean_ci_summary(
        plot_df,
        group_columns=[
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
        metric_columns=available_metrics,
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )
    summary["metricLabel"] = summary["metric"].map(
        AI_EXPERIENCE_METRICS
    )

    save_table(summary, slug, index=False)

    # Separate the three workflow estimates around each round position.
    workflow_offsets = dict(
        zip(
            ai_workflows,
            np.linspace(-0.14, 0.14, len(ai_workflows)),
        )
    )

    fig, axes = plt.subplots(
        len(available_metrics),
        len(groups),
        figsize=(
            6.3 * len(groups),
            3.55 * len(available_metrics),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    round_labels = [
        (
            f"Main round {index + 1}\nInjected error"
            if round_index == ERROR_ROUND_INDEX
            else f"Main round {index + 1}"
        )
        for index, round_index in enumerate(rounds)
    ]

    for row_index, metric in enumerate(available_metrics):
        for column_index, group in enumerate(groups):
            axis = axes[row_index, column_index]

            panel_summary = summary.loc[
                summary["errorExposed"].eq(group)
                & summary["metric"].eq(metric)
                ]

            for workflow in ai_workflows:
                workflow_summary = (
                    panel_summary.loc[
                        panel_summary["workflow"].eq(workflow)
                    ]
                    .set_index("roundIndex")
                    .reindex(rounds)
                )

                valid = workflow_summary["mean"].notna().to_numpy()
                if not valid.any():
                    continue

                x_values = (
                        np.asarray(rounds, dtype=float)
                        + workflow_offsets[workflow]
                )
                means = workflow_summary["mean"].to_numpy(dtype=float)
                counts = (
                    pd.to_numeric(
                        workflow_summary["count"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .to_numpy(dtype=int)
                )

                # Separate points without lines between rounds.
                axis.scatter(
                    x_values[valid],
                    means[valid],
                    s=46,
                    color=WORKFLOW_COLORS[workflow],
                    edgecolor="white",
                    linewidth=0.7,
                    label=workflow_display_name(workflow),
                    zorder=4,
                )

                # Show confidence intervals only for cells with at least
                # three observations.
                ci_valid = (
                        workflow_summary["mean"].notna()
                        & workflow_summary["lowerCI"].notna()
                        & workflow_summary["upperCI"].notna()
                        & workflow_summary["count"].ge(3)
                ).to_numpy()

                if ci_valid.any():
                    ci_rows = workflow_summary.loc[
                        workflow_summary["mean"].notna()
                        & workflow_summary["lowerCI"].notna()
                        & workflow_summary["upperCI"].notna()
                        & workflow_summary["count"].ge(3)
                        ]

                    lower_ci = ci_rows["lowerCI"].to_numpy(dtype=float)
                    upper_ci = ci_rows["upperCI"].to_numpy(dtype=float)

                    axis.errorbar(
                        x_values[ci_valid],
                        means[ci_valid],
                        yerr=np.vstack(
                            [
                                means[ci_valid] - lower_ci,
                                upper_ci - means[ci_valid],
                                ]
                        ),
                        fmt="none",
                        ecolor=WORKFLOW_COLORS[workflow],
                        elinewidth=1.05,
                        capsize=3,
                        alpha=0.9,
                        zorder=3,
                    )

                # Display the sample size for every observed cell.
                for x_value, mean, count in zip(
                        x_values[valid],
                        means[valid],
                        counts[valid],
                ):
                    count_label = f"n={count}"

                    axis.annotate(
                        count_label,
                        (x_value, mean),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.7,
                        color=WORKFLOW_COLORS[workflow],
                    )

            axis.set_xticks(rounds)
            axis.set_xticklabels(round_labels)
            axis.set_ylim(0.7, 5.42)
            axis.yaxis.set_major_locator(
                MaxNLocator(integer=True)
            )
            axis.set_xlabel("Main round")

            if column_index == 0:
                axis.set_ylabel(
                    f"{AI_EXPERIENCE_METRICS[metric]}\n"
                    "Mean rating (1–5)"
                )

            if row_index == 0:
                axis.set_title(
                    exposure_display_name(group)
                )

            apply_standard_axes_style(
                axis,
                grid_axis="y",
            )

    # Collect unique legend entries from all panels.
    legend_items = {}

    for axis in axes.flatten():
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="AI supported workflow",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "AI Interaction Ratings by Main Round, Workflow, and Error Exposure",
        fontsize=13,
        y=0.995,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate Main round and workflow means and are not "
            "connected because participants could switch workflows between "
            "rounds. Error bars are shown for cells with at least three "
            "observations."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.045, 0.84, 0.97)
    )

    save_figure(
        fig,
        slug,
        "AI Interaction Ratings by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean AI interaction ratings for each Main round, "
            "AI supported workflow, and actual error exposure group. Points "
            "are not connected because participants could switch workflows "
            "between Main rounds. Error bars show approximate 95% confidence "
            "intervals for cells with at least three observations."
        ),
    )


# ---------------------------------------------------------------------------
# 104: Raw NASA-TLX workload in Main rounds
# ---------------------------------------------------------------------------


def plot_main_round_tlx_by_exposure_and_workflow(
        main_df: pd.DataFrame,
) -> None:
    """Show Raw NASA-TLX by Main round, error exposure, and selected workflow.

    Each point represents a separate exposure, Main round, and workflow cell.
    Points are not connected because participants could switch workflows between
    Main rounds.
    """
    slug = "104_main_round_tlx_by_exposure_round_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "rawNasaTlxScore",
    }
    if not require_columns(
            main_df,
            required,
            "Raw NASA-TLX in Main Rounds by error exposure",
    ):
        return

    plot_df = main_df[
        [
            "roundIndex",
            "workflow",
            "errorExposed",
            "rawNasaTlxScore",
        ]
    ].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )
    plot_df["rawNasaTlxScore"] = pd.to_numeric(
        plot_df["rawNasaTlxScore"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            "rawNasaTlxScore",
        ]
    )
    plot_df = plot_df.loc[
        plot_df["workflow"].isin(WORKFLOW_ORDER)
        & plot_df["rawNasaTlxScore"].between(0, 20)
        ].copy()

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)
    rounds = sorted(plot_df["roundIndex"].unique().tolist())

    if not groups or not rounds:
        return

    summary = _mean_ci_summary(
        plot_df,
        group_columns=[
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
        metric_columns=["rawNasaTlxScore"],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )

    save_table(summary, slug, index=False)

    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(
                -0.18,
                0.18,
                len(WORKFLOW_ORDER),
            ),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.7 * len(groups), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(group)
            & summary["metric"].eq("rawNasaTlxScore")
            ]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            valid = workflow_summary["mean"].notna().to_numpy()
            if not valid.any():
                continue

            x_values = (
                    np.asarray(rounds, dtype=float)
                    + workflow_offsets[workflow]
            )
            means = workflow_summary["mean"].to_numpy(dtype=float)

            counts = (
                pd.to_numeric(
                    workflow_summary["count"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            # Separate points without connecting lines.
            axis.scatter(
                x_values[valid],
                means[valid],
                s=52,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.8,
                label=workflow_display_name(workflow),
                zorder=4,
            )

            # Confidence intervals are shown only when at least three
            # observations support the cell estimate.
            ci_valid = (
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna()
                    & workflow_summary["count"].ge(3)
            ).to_numpy()

            if ci_valid.any():
                ci_rows = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna()
                    & workflow_summary["count"].ge(3)
                    ]

                lower_ci = np.clip(
                    ci_rows["lowerCI"].to_numpy(dtype=float),
                    0,
                    20,
                )
                upper_ci = np.clip(
                    ci_rows["upperCI"].to_numpy(dtype=float),
                    0,
                    20,
                )

                axis.errorbar(
                    x_values[ci_valid],
                    means[ci_valid],
                    yerr=np.vstack(
                        [
                            means[ci_valid] - lower_ci,
                            upper_ci - means[ci_valid],
                            ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.1,
                    capsize=3,
                    alpha=0.9,
                    zorder=3,
                )

            for x_value, mean, count in zip(
                    x_values[valid],
                    means[valid],
                    counts[valid],
            ):
                count_label = f"n={count}"

                axis.annotate(
                    count_label,
                    (x_value, mean),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                )

        round_labels = [
            (
                f"Main round {index + 1}\nInjected error"
                if round_index == ERROR_ROUND_INDEX
                else f"Main round {index + 1}"
            )
            for index, round_index in enumerate(rounds)
        ]

        axis.set_xticks(rounds)
        axis.set_xticklabels(round_labels)
        axis.set_xlim(
            min(rounds) - 0.45,
            max(rounds) + 0.45,
            )
        axis.set_ylim(-0.5, 20.8)
        axis.set_yticks([0, 5, 10, 15, 20])

        axis.set_title(
            exposure_display_name(group)
        )
        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel(
        "Raw NASA-TLX workload score (0–20)"
    )

    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "Raw NASA-TLX Workload in Main Rounds by Error Exposure",
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate Main round and workflow means and are not "
            "connected because participants could switch workflows between "
            "rounds. Error bars are shown for cells with at least three "
            "observations. Higher scores indicate greater perceived workload."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.065, 0.84, 0.96)
    )

    save_figure(
        fig,
        slug,
        "Raw NASA-TLX Workload in Main Rounds by Error Exposure",
        (
            "Separate mean Raw NASA-TLX estimates for each Main round, selected "
            "workflow, and actual error exposure group. Points are not connected "
            "because workflow-group membership could change between Main rounds. "
            "Error bars show approximate 95% confidence intervals for cells with "
            "at least three observations."
        ),
    )


# ---------------------------------------------------------------------------
# 105: Post-error workflow choices by Main Round 1 exposure
# ---------------------------------------------------------------------------


def plot_post_error_workflow_choices_by_exposure(prepared) -> None:
    """Show post-error Main Round 2-3 workflow distributions by Round-5 exposure."""
    slug = "105_post_error_workflow_choices_by_exposure"

    post = (
        prepared[prepared["roundIndex"].gt(ERROR_ROUND_INDEX)]
        .dropna(subset=["errorExposed"])
        .copy()
    )
    if post.empty:
        return

    groups = [group for group in [True, False] if group in set(post["errorExposed"])]
    rounds = sorted(post["roundIndex"].unique().tolist())
    grid = pd.MultiIndex.from_product(
        [groups, rounds, WORKFLOW_ORDER],
        names=["errorExposed", "roundIndex", "workflow"],
    )
    summary = (
        post.groupby(["errorExposed", "roundIndex", "workflow"])
        .size()
        .reindex(grid, fill_value=0)
        .rename("choiceCount")
        .reset_index()
    )
    summary["roundTotal"] = summary.groupby(["errorExposed", "roundIndex"])[
        "choiceCount"
    ].transform("sum")
    summary["choicePercentage"] = np.where(
        summary["roundTotal"] > 0,
        summary["choiceCount"] / summary["roundTotal"] * 100,
        np.nan,
        )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    summary["mainRoundLabel"] = summary["roundIndex"].map(round_display_name)
    save_table(summary, slug, index=False)

    fig, axes = plt.subplots(
        1, len(groups), figsize=(6.2 * len(groups), 5.5), sharey=True, squeeze=False
    )
    for ax, group in zip(axes.flatten(), groups):
        group_summary = summary[summary["errorExposed"].eq(group)]
        bottoms = np.zeros(len(rounds), dtype=float)

        for workflow in WORKFLOW_ORDER:
            values = (
                group_summary[group_summary["workflow"].eq(workflow)]
                .set_index("roundIndex")
                .reindex(rounds)
            )
            percentages = values["choicePercentage"].to_numpy(dtype=float)
            counts = values["choiceCount"].to_numpy(dtype=int)
            bars = ax.bar(
                np.arange(len(rounds)),
                percentages,
                bottom=bottoms,
                color=WORKFLOW_COLORS[workflow],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.8,
                label=workflow_display_name(workflow),
                zorder=2,
            )
            for bar, percent, count, bottom in zip(bars, percentages, counts, bottoms):
                if percent >= 9:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + percent / 2,
                        f"{count}\n{percent:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="black",
                        )
            bottoms += percentages

        totals = (
            group_summary.drop_duplicates("roundIndex")
            .set_index("roundIndex")
            .reindex(rounds)["roundTotal"]
            .to_numpy(dtype=int)
        )
        ax.set_xticks(np.arange(len(rounds)))
        ax.set_xticklabels(
            [f"{round_display_name(r)}\nn={n}" for r, n in zip(rounds, totals)]
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("Post-error in main rounds")
        ax.set_title(exposure_display_name(group))
        apply_standard_axes_style(ax, grid_axis="y")

    axes[0, 0].set_ylabel("Workflow choices (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Workflow selected",
        bbox_to_anchor=(0.99, 0.5),
        loc="center left",
    )
    fig.suptitle(
        "Post-Error Workflow Choices by Main Round 1 Exposure", fontsize=13, y=0.99
    )
    fig.text(
        0.01,
        0.01,
        "Each bar contains all workflow choices in that round. Exposure was defined from Main Round 1 and was not independently randomized.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Main Round 1 Exposure",
        "Distribution of voluntary workflow choices in Main Rounds 2-3 by error exposure.",
    )


# ---------------------------------------------------------------------------
# 106: Main round quality patterns by AI-error exposure
# ---------------------------------------------------------------------------


def plot_main_round_quality_by_error_exposure(
        main_df: pd.DataFrame,
) -> None:
    """Show Main round quality by exposure group and selected workflow.

    Each point represents a separate exposure-group, Main-round, and workflow
    mean. Points are not connected because participants could change workflows
    between Main rounds.
    """
    slug = "106_main_round_quality_by_error_exposure"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
    }
    if not require_columns(
            main_df,
            required,
            "Main round quality by exposure",
    ):
        return

    plot_df = main_df.copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
        ]
    )

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
        ].copy()

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)

    rounds = [
        round_index
        for round_index in MAIN_ROUND_INDICES
        if round_index in plot_df["roundIndex"].unique()
    ]

    if not groups or not rounds:
        return

    summary = quality_summary(
        plot_df,
        [
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
    )

    if summary.empty:
        return

    # Add counts separately if quality_summary does not already return them.
    if "count" not in summary.columns:
        counts = (
            plot_df
            .groupby(
                [
                    "errorExposed",
                    "roundIndex",
                    "workflow",
                ],
                dropna=False,
            )
            .size()
            .rename("count")
            .reset_index()
        )

        summary = summary.merge(
            counts,
            on=[
                "errorExposed",
                "roundIndex",
                "workflow",
            ],
            how="left",
        )

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )

    save_table(summary, slug, index=False)

    # Separate workflow estimates slightly around each Main-round position.
    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(
                -0.18,
                0.18,
                len(WORKFLOW_ORDER),
            ),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.6 * len(groups), 5.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, exposed in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(exposed)
        ]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            means = pd.to_numeric(
                workflow_summary["mean"],
                errors="coerce",
            ).to_numpy(dtype=float)

            lows = pd.to_numeric(
                workflow_summary["ciLow"],
                errors="coerce",
            ).to_numpy(dtype=float)

            highs = pd.to_numeric(
                workflow_summary["ciHigh"],
                errors="coerce",
            ).to_numpy(dtype=float)

            counts = (
                pd.to_numeric(
                    workflow_summary["count"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            valid = np.isfinite(means)

            if not valid.any():
                continue

            x_values = (
                    np.asarray(rounds, dtype=float)
                    + workflow_offsets[workflow]
            )

            # Mean dots without connecting lines.
            axis.scatter(
                x_values[valid],
                means[valid],
                s=52,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.8,
                label=workflow_display_name(workflow),
                zorder=4,
            )

            valid_ci = (
                    valid
                    & np.isfinite(lows)
                    & np.isfinite(highs)
            )

            if valid_ci.any():
                clipped_lows = np.clip(
                    lows[valid_ci],
                    QUALITY_Y_MIN,
                    QUALITY_Y_MAX,
                )
                clipped_highs = np.clip(
                    highs[valid_ci],
                    QUALITY_Y_MIN,
                    QUALITY_Y_MAX,
                )

                axis.errorbar(
                    x_values[valid_ci],
                    means[valid_ci],
                    yerr=np.vstack(
                        [
                            means[valid_ci] - clipped_lows,
                            clipped_highs - means[valid_ci],
                            ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.1,
                    capsize=3,
                    capthick=1.0,
                    alpha=0.9,
                    zorder=3,
                )

            # Add the sample size above each point.
            for x_value, mean, count in zip(
                    x_values[valid],
                    means[valid],
                    counts[valid],
            ):
                axis.annotate(
                    f"n={count}",
                    (x_value, mean),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                    zorder=5,
                )

        # Main 1 is the round in which exposure status was determined.
        axis.axvline(
            ERROR_ROUND_INDEX,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.55,
            zorder=1,
        )

        axis.set_title(
            exposure_display_name(exposed)
        )

        axis.set_xticks(rounds)
        axis.set_xticklabels(
            _main_round_labels(rounds)
        )
        axis.set_xlim(
            min(rounds) - 0.45,
            max(rounds) + 0.45,
            )

        axis.set_ylim(
            QUALITY_Y_MIN - 0.1,
            QUALITY_Y_MAX + 0.35,
            )

        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel(
        "Mean overall quality (1–5)"
    )

    # Build one shared legend from all panels because some workflows may
    # not appear in one of the exposure groups.
    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "Mean Overall Quality by Main Round, Workflow, and Error Exposure",
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate Main-round and workflow means and are not "
            "connected because participants could switch workflows between "
            "rounds. Whiskers show approximate 95% confidence intervals. "
            "The dashed line marks Main 1, when exposure status was determined. "
            "Cells with very small sample sizes should be interpreted cautiously."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.065, 0.84, 0.96)
    )

    save_figure(
        fig,
        slug,
        "Mean Overall Quality by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean overall-quality estimates for each Main round, "
            "selected workflow, and actual error-exposure group. Points are "
            "not connected because workflow membership could change between "
            "rounds. Whiskers show approximate 95% confidence intervals and "
            "labels show observation counts."
        ),
    )


# ---------------------------------------------------------------------------
# 107: Complete constraint fulfillment across Main rounds by exposure
# ---------------------------------------------------------------------------


def plot_main_constraint_fulfillment_by_exposure(
        main_df: pd.DataFrame,
) -> None:
    """Show constraint-fulfillment rates by exposure, round, and workflow.

    Each point represents one exposure-group, Main-round, and selected-workflow
    cell. Points are not connected because participants could switch workflows
    between Main rounds. Whiskers show 95% Wilson confidence intervals.
    """
    slug = (
        "107_main_constraint_fulfillment_by_error_exposure_round_and_workflow"
    )

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "passedNumeric",
    }
    if not require_columns(
            main_df,
            required,
            "Main round constraint fulfillment by exposure and workflow",
    ):
        return

    plot_df = main_df[
        [
            "roundIndex",
            "workflow",
            "errorExposed",
            "passedNumeric",
        ]
    ].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )
    plot_df["passedNumeric"] = pd.to_numeric(
        plot_df["passedNumeric"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            "passedNumeric",
        ]
    )

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
        & plot_df["passedNumeric"].isin([0, 1])
        ].copy()

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)

    main_rounds = [
        round_index
        for round_index in MAIN_ROUND_INDICES
        if round_index in plot_df["roundIndex"].unique()
    ]

    if not groups or not main_rounds:
        return

    summary = pass_summary(
        plot_df,
        [
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
    )

    if summary.empty:
        return

    counts = (
        plot_df
        .groupby(
            [
                "errorExposed",
                "roundIndex",
                "workflow",
            ],
            dropna=False,
        )
        .agg(
            passedCount=("passedNumeric", "sum"),
            observedCount=("passedNumeric", "size"),
        )
        .reset_index()
    )

    summary = summary.merge(
        counts,
        on=[
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
        how="left",
    )

    if "totalRounds" not in summary.columns:
        summary["totalRounds"] = summary["observedCount"]

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )

    save_table(summary, slug, index=False)

    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(
                -0.18,
                0.18,
                len(WORKFLOW_ORDER),
            ),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.7 * len(groups), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(group)
        ]

        # Main 1 is the exposure-defining round.
        if ERROR_ROUND_INDEX in main_rounds:
            axis.axvspan(
                ERROR_ROUND_INDEX - 0.32,
                ERROR_ROUND_INDEX + 0.32,
                facecolor="#F1F1F1",
                edgecolor="none",
                zorder=0,
                )

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(main_rounds)
            )

            rates = pd.to_numeric(
                workflow_summary["passRatePercent"],
                errors="coerce",
            ).to_numpy(dtype=float)

            lower_ci = pd.to_numeric(
                workflow_summary["lowerCI"],
                errors="coerce",
            ).to_numpy(dtype=float)

            upper_ci = pd.to_numeric(
                workflow_summary["upperCI"],
                errors="coerce",
            ).to_numpy(dtype=float)

            passed_counts = (
                pd.to_numeric(
                    workflow_summary["passedCount"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            total_counts = (
                pd.to_numeric(
                    workflow_summary["totalRounds"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            valid = np.isfinite(rates)

            if not valid.any():
                continue

            x_values = (
                    np.asarray(main_rounds, dtype=float)
                    + workflow_offsets[workflow]
            )

            valid_ci = (
                    valid
                    & np.isfinite(lower_ci)
                    & np.isfinite(upper_ci)
            )

            if valid_ci.any():
                clipped_lower = np.clip(
                    lower_ci[valid_ci],
                    0,
                    100,
                )
                clipped_upper = np.clip(
                    upper_ci[valid_ci],
                    0,
                    100,
                )

                axis.errorbar(
                    x_values[valid_ci],
                    rates[valid_ci],
                    yerr=np.vstack(
                        [
                            rates[valid_ci] - clipped_lower,
                            clipped_upper - rates[valid_ci],
                            ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.25,
                    capsize=4,
                    capthick=1.1,
                    alpha=0.9,
                    zorder=2,
                )

            # Separate estimates without connecting lines.
            axis.scatter(
                x_values[valid],
                rates[valid],
                s=66,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.9,
                label=workflow_display_name(workflow),
                zorder=3,
            )

            for x_value, rate, passed, total in zip(
                    x_values[valid],
                    rates[valid],
                    passed_counts[valid],
                    total_counts[valid],
            ):
                small_sample = total < 3
                count_label = f"{passed}/{total}"

                if rate >= 82:
                    text_offset = (0, -11)
                    vertical_alignment = "top"
                else:
                    text_offset = (0, 8)
                    vertical_alignment = "bottom"

                axis.annotate(
                    count_label,
                    (x_value, rate),
                    xytext=text_offset,
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                    zorder=4,
                )

        group_is_exposed = str(group).strip().lower() in {
            "true",
            "1",
            "yes",
            "exposed",
            "error-exposed",
        }

        panel_note = (
            "AI-supported workflow selected in Main 1;\n"
            "line-count error injected"
            if group_is_exposed
            else
            "Human-only selected in Main 1;\n"
            "no AI error injected"
        )

        axis.set_title(
            exposure_display_name(group),
            fontsize=11.5,
            pad=34,
        )

        axis.text(
            0.5,
            1.025,
            panel_note,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#555555",
        )

        round_labels = []

        for index, round_index in enumerate(main_rounds):
            label = f"Main {index + 1}"

            if round_index == ERROR_ROUND_INDEX:
                label += "\nExposure-defining round"

            round_labels.append(label)

        axis.set_xticks(main_rounds)
        axis.set_xticklabels(round_labels)

        axis.set_xlim(
            min(main_rounds) - 0.45,
            max(main_rounds) + 0.45,
            )
        axis.set_ylim(-4, 108)
        axis.set_yticks([0, 20, 40, 60, 80, 100])

        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel(
        "Rounds fulfilling all constraints (%)"
    )

    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        (
            "Complete Constraint Fulfillment by Main Round, "
            "Workflow, and Error Exposure"
        ),
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate exposure-group, Main-round, and workflow "
            "pass-rate estimates. Whiskers show 95% Wilson confidence "
            "intervals, and labels show passed/observed rounds. Points are not "
            "connected because participants could switch workflows between "
            "rounds. Main 1 determined exposure status; "
            "workflow choice was unrestricted again in Main 2 and Main 3."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.09, 0.84, 0.94)
    )

    save_figure(
        fig,
        slug,
        (
            "Complete Constraint Fulfillment by Main Round, "
            "Workflow, and Error Exposure"
        ),
        (
            "Complete constraint-fulfillment rates for each Main round and "
            "selected workflow, shown separately by actual Main 1 error "
            "exposure. Points show pass-rate estimates, whiskers show 95% "
            "Wilson confidence intervals, and labels report successful rounds "
            "divided by observed rounds."
        ),
    )


# ---------------------------------------------------------------------------
# 108: Line-count failure pattern across Main rounds by exposure
# ---------------------------------------------------------------------------


def _line_count_error(value) -> bool | None:
    """Return whether the single line-count rule failed for one round."""
    for item in parse_requirement_results(value):
        rule_id = str(item.get("id", ""))
        if not rule_id.startswith("lines-"):
            continue

        passed = parse_bool_or_none(item.get("passed"))
        return None if passed is None else not passed

    return None


def plot_main_line_count_error_by_exposure(
        main_df: pd.DataFrame,
) -> None:
    """Show line-count failure rates by exposure, Main round, and workflow.

    Each point represents one exposure-group, Main-round, and selected-workflow
    cell. Points are not connected because participants could switch workflows
    between Main rounds. Whiskers show 95% Wilson confidence intervals.
    """
    slug = (
        "108_main_line_count_error_by_error_exposure_round_and_workflow"
    )

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "requirementResults",
    }
    if not require_columns(
            main_df,
            required,
            "Main round line-count failures by exposure and workflow",
    ):
        return

    plot_df = main_df[
        [
            "roundIndex",
            "workflow",
            "errorExposed",
            "requirementResults",
        ]
    ].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    plot_df["lineCountError"] = plot_df[
        "requirementResults"
    ].apply(_line_count_error)

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            "lineCountError",
        ]
    )

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)
    plot_df["lineCountError"] = (
        plot_df["lineCountError"].astype(bool)
    )

    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
        ].copy()

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)

    main_rounds = [
        round_index
        for round_index in MAIN_ROUND_INDICES
        if round_index in plot_df["roundIndex"].unique()
    ]

    if not groups or not main_rounds:
        return

    summary = (
        plot_df
        .groupby(
            [
                "errorExposed",
                "roundIndex",
                "workflow",
            ],
            dropna=False,
        )["lineCountError"]
        .agg(
            totalRounds="count",
            failedLineCount="sum",
        )
        .reset_index()
    )

    summary["totalRounds"] = pd.to_numeric(
        summary["totalRounds"],
        errors="coerce",
    ).astype(int)

    summary["failedLineCount"] = pd.to_numeric(
        summary["failedLineCount"],
        errors="coerce",
    ).astype(int)

    summary["lineCountFailureRatePercent"] = (
            summary["failedLineCount"]
            / summary["totalRounds"]
            * 100
    )

    intervals = summary.apply(
        lambda row: wilson_interval(
            int(row["failedLineCount"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )

    summary[["lowerCI", "upperCI"]] = intervals

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["exposureLabel"] = summary["errorExposed"].map(
        exposure_display_name
    )

    save_table(summary, slug, index=False)

    # Slightly separate the workflows around each Main-round position.
    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(
                -0.18,
                0.18,
                len(WORKFLOW_ORDER),
            ),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.7 * len(groups), 5.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(group)
        ]

        # Main 1 is both the manipulation round and the round that defines
        # exposure-group membership.
        if ERROR_ROUND_INDEX in main_rounds:
            axis.axvspan(
                ERROR_ROUND_INDEX - 0.32,
                ERROR_ROUND_INDEX + 0.32,
                facecolor="#F1F1F1",
                edgecolor="none",
                zorder=0,
                )

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(main_rounds)
            )

            rates = pd.to_numeric(
                workflow_summary[
                    "lineCountFailureRatePercent"
                ],
                errors="coerce",
            ).to_numpy(dtype=float)

            lower_ci = pd.to_numeric(
                workflow_summary["lowerCI"],
                errors="coerce",
            ).to_numpy(dtype=float)

            upper_ci = pd.to_numeric(
                workflow_summary["upperCI"],
                errors="coerce",
            ).to_numpy(dtype=float)

            failed_counts = (
                pd.to_numeric(
                    workflow_summary["failedLineCount"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            total_counts = (
                pd.to_numeric(
                    workflow_summary["totalRounds"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            valid = np.isfinite(rates)

            if not valid.any():
                continue

            x_values = (
                    np.asarray(main_rounds, dtype=float)
                    + workflow_offsets[workflow]
            )

            valid_ci = (
                    valid
                    & np.isfinite(lower_ci)
                    & np.isfinite(upper_ci)
            )

            if valid_ci.any():
                clipped_lower = np.clip(
                    lower_ci[valid_ci],
                    0,
                    100,
                )
                clipped_upper = np.clip(
                    upper_ci[valid_ci],
                    0,
                    100,
                )

                axis.errorbar(
                    x_values[valid_ci],
                    rates[valid_ci],
                    yerr=np.vstack(
                        [
                            rates[valid_ci] - clipped_lower,
                            clipped_upper - rates[valid_ci],
                            ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.25,
                    capsize=4,
                    capthick=1.1,
                    alpha=0.9,
                    zorder=2,
                )

            # Separate cell estimates without connecting lines.
            axis.scatter(
                x_values[valid],
                rates[valid],
                s=66,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.9,
                label=workflow_display_name(workflow),
                zorder=3,
            )

            for x_value, rate, failed, total in zip(
                    x_values[valid],
                    rates[valid],
                    failed_counts[valid],
                    total_counts[valid],
            ):
                label = f"{failed}/{total}"

                # Keep labels inside the plot and away from the whiskers.
                if rate >= 82:
                    offset = (0, -11)
                    vertical_alignment = "top"
                else:
                    offset = (0, 8)
                    vertical_alignment = "bottom"

                axis.annotate(
                    label,
                    (x_value, rate),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                    zorder=4,
                )

        group_is_exposed = (
            bool(group)
            if isinstance(group, (bool, np.bool_))
            else str(group).strip().lower().replace("_", "-")
                 in {
                     "true",
                     "1",
                     "yes",
                     "exposed",
                     "error-exposed",
                 }
        )

        panel_note = (
            "AI-supported workflow selected in Main 1;\n"
            "line-count error injected"
            if group_is_exposed
            else
            "Human-only selected in Main 1;\n"
            "no AI error injected"
        )

        axis.set_title(
            exposure_display_name(group),
            fontsize=11.5,
            pad=34,
        )

        axis.text(
            0.5,
            1.025,
            panel_note,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#555555",
        )

        round_labels = []

        for index, round_index in enumerate(main_rounds):
            label = f"Main {index + 1}"

            if round_index == ERROR_ROUND_INDEX:
                label += "\nInjected-error round"

            round_labels.append(label)

        axis.set_xticks(main_rounds)
        axis.set_xticklabels(round_labels)

        axis.set_xlim(
            min(main_rounds) - 0.45,
            max(main_rounds) + 0.45,
            )
        axis.set_ylim(-4, 108)
        axis.set_yticks([0, 20, 40, 60, 80, 100])

        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel(
        "Rounds with a line-count failure (%)"
    )

    # Build one shared workflow legend across both exposure panels.
    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        (
            "Line-Count Failure Rate by Main Round, "
            "Workflow, and Error Exposure"
        ),
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate exposure-group, Main-round, and workflow "
            "line-count failure rates. Whiskers show 95% Wilson confidence "
            "intervals, and labels show failed/observed rounds. Points are not "
            "connected because participants could switch workflows between "
            "rounds. The injected line-count error occurred only in Main 1 "
            "for participants selecting an AI-supported workflow."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.095, 0.84, 0.94)
    )

    save_figure(
        fig,
        slug,
        (
            "Line-Count Failure Rate by Main Round, "
            "Workflow, and Error Exposure"
        ),
        (
            "Line-count failure rates for each Main round and selected workflow, "
            "shown separately by actual Main 1 error exposure. Points show "
            "failure-rate estimates, whiskers show 95% Wilson confidence "
            "intervals, and labels report failed rounds divided by observed "
            "rounds."
        ),
    )

# ---------------------------------------------------------------------------
# 109: Final workflow preference by reported AI errors
# ---------------------------------------------------------------------------


def _reported_ai_error_groups(
        round_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one reported-AI-error group per participant session."""
    notes = load_participant_interview_notes(round_df)

    if notes.empty or "sessionId" not in notes.columns:
        return pd.DataFrame()

    notes = notes.dropna(subset=["sessionId"]).copy()
    notes["sessionId"] = notes["sessionId"].astype(str)

    injected_error_count = (
        notes["injectedErrorExperience"].eq("noticed").astype(int)
        if "injectedErrorExperience" in notes.columns
        else pd.Series(0, index=notes.index, dtype=int)
    )

    if "reportedOtherAiErrorTypes" in notes.columns:
        other_error_counts = (
            notes[["participantId", "reportedOtherAiErrorTypes"]]
            .dropna(subset=["reportedOtherAiErrorTypes"])
            .assign(
                errorType=lambda data: (
                    data["reportedOtherAiErrorTypes"]
                    .astype("string")
                    .str.lower()
                    .str.split(";")
                )
            )
            .explode("errorType")
            .assign(errorType=lambda data: data["errorType"].str.strip())
            .loc[lambda data: data["errorType"].notna() & data["errorType"].ne("")]
            .groupby("participantId")["errorType"]
            .nunique()
        )

        notes["otherAiErrorCount"] = (
            notes["participantId"].map(other_error_counts).fillna(0).astype(int)
        )
    else:
        notes["otherAiErrorCount"] = 0

    notes["reportedAiErrorCount"] = injected_error_count + notes["otherAiErrorCount"]

    notes["errorGroup"] = pd.cut(
        notes["reportedAiErrorCount"],
        bins=[-1, 0, 1, np.inf],
        labels=[
            "No reported AI errors",
            "1 reported AI error",
            "2+ reported AI errors",
        ],
    )

    return notes[
        [
            "sessionId",
            "reportedAiErrorCount",
            "errorGroup",
        ]
    ].dropna(subset=["errorGroup"])


def plot_final_workflow_preference_by_reported_ai_errors(
        ranking_rows: pd.DataFrame,
        prepared: pd.DataFrame,
) -> None:
    """Compare final workflow preferences by reported AI-error count.

    The left panel shows average assigned rank. The right panel shows the
    percentage of participants who ranked each workflow first.

    Lower mean ranks indicate stronger preference. Reported-error groups are
    descriptive and do not necessarily represent actual error exposure.
    """
    from matplotlib.lines import Line2D

    slug = "109_final_workflow_preference_by_reported_ai_errors"

    error_groups = _reported_ai_error_groups(prepared)

    if ranking_rows.empty or error_groups.empty:
        return

    ranking_df = ranking_rows.copy()
    ranking_df["sessionId"] = ranking_df["sessionId"].astype(str)

    error_groups = error_groups.copy()
    error_groups["sessionId"] = error_groups["sessionId"].astype(str)

    ranking_df = ranking_df.merge(
        error_groups,
        on="sessionId",
        how="inner",
        validate="many_to_one",
    )

    if ranking_df.empty:
        return

    group_order = [
        "No reported AI errors",
        "1 reported AI error",
        "2+ reported AI errors",
    ]

    observed_groups = [
        group
        for group in group_order
        if group in set(ranking_df["errorGroup"])
    ]

    if not observed_groups:
        return

    rank_columns = list(
        range(1, len(WORKFLOW_ORDER) + 1)
    )

    # Keep the workflow order identical in both panels and all groups.
    overall_summary = ranking_summary(ranking_df)

    workflow_order = (
        overall_summary["meanRank"]
        .sort_values()
        .index
        .tolist()
    )

    if not workflow_order:
        return

    group_sizes = {
        group: ranking_df.loc[
            ranking_df["errorGroup"].eq(group),
            "sessionId",
        ].nunique()
        for group in observed_groups
    }

    group_colors = {
        "No reported AI errors": "#4C78A8",
        "1 reported AI error": "#F58518",
        "2+ reported AI errors": "#B279A2",
    }

    group_markers = {
        "No reported AI errors": "o",
        "1 reported AI error": "s",
        "2+ reported AI errors": "^",
    }

    group_offsets = dict(
        zip(
            observed_groups,
            np.linspace(
                -0.20,
                0.20,
                len(observed_groups),
            ),
        )
    )

    group_summaries = {}
    export_rows = []

    for group in observed_groups:
        group_df = ranking_df.loc[
            ranking_df["errorGroup"].eq(group)
        ]

        summary = (
            ranking_summary(group_df)
            .reindex(workflow_order)
        )

        rank_counts = (
            summary[rank_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        row_totals = rank_counts.sum(axis=1)

        first_choice_percent = (
            rank_counts[1]
            .div(row_totals.replace(0, np.nan))
            .mul(100)
        )

        summary = summary.copy()
        summary["firstChoicePercent"] = first_choice_percent
        summary["rankingCount"] = row_totals

        group_summaries[group] = summary

        for workflow in workflow_order:
            mean_rank = summary.loc[workflow, "meanRank"]
            ranking_count = summary.loc[
                workflow,
                "rankingCount",
            ]
            first_choice_percentage = summary.loc[
                workflow,
                "firstChoicePercent",
            ]

            row = {
                "errorGroup": group,
                "groupParticipants": group_sizes[group],
                "workflow": workflow_display_name(workflow),
                "meanRank": mean_rank,
                "firstChoicePercent": first_choice_percentage,
                "validRankings": int(ranking_count),
            }

            for rank in rank_columns:
                count = int(rank_counts.loc[workflow, rank])

                percentage = (
                    count / ranking_count * 100
                    if ranking_count > 0
                    else np.nan
                )

                row[f"Rank {rank} count"] = count
                row[f"Rank {rank} percent"] = percentage

            export_rows.append(row)

    save_table(
        pd.DataFrame(export_rows),
        slug,
        index=False,
    )

    positions = np.arange(len(workflow_order))

    fig, (ax_mean, ax_first) = plt.subplots(
        ncols=2,
        figsize=(12.4, 5.5),
        sharey=True,
        gridspec_kw={
            "width_ratios": [1.15, 1.35],
        },
    )

    # ------------------------------------------------------------
    # Left panel: average preference rank
    # ------------------------------------------------------------
    for group in observed_groups:
        summary = group_summaries[group]
        offset = group_offsets[group]
        color = group_colors[group]
        marker = group_markers[group]

        mean_ranks = pd.to_numeric(
            summary["meanRank"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(mean_ranks)
        y_values = positions + offset

        # Subtle lollipop stems, without implying a trajectory.
        for y_value, mean_rank in zip(
                y_values[valid],
                mean_ranks[valid],
        ):
            ax_mean.hlines(
                y=y_value,
                xmin=1,
                xmax=mean_rank,
                color=color,
                linewidth=1.2,
                alpha=0.35,
                zorder=1,
            )

        ax_mean.scatter(
            mean_ranks[valid],
            y_values[valid],
            s=78,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

        for mean_rank, y_value in zip(
                mean_ranks[valid],
                y_values[valid],
        ):
            ax_mean.annotate(
                f"{mean_rank:.2f}",
                (mean_rank, y_value),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=color,
            )

    ax_mean.set_yticks(positions)
    ax_mean.set_yticklabels(
        [
            (
                f"{index + 1}. "
                f"{workflow_display_name(workflow)}"
            )
            for index, workflow in enumerate(workflow_order)
        ]
    )
    ax_mean.invert_yaxis()

    ax_mean.set_xlim(0.8, 4.45)
    ax_mean.set_xticks([1, 2, 3, 4])
    ax_mean.set_xticklabels(
        [
            "1\nBest",
            "2",
            "3",
            "4\nWorst",
        ]
    )

    ax_mean.set_xlabel("Average assigned rank")
    ax_mean.set_title("Average preference rank")

    apply_standard_axes_style(
        ax_mean,
        grid_axis="x",
    )

    # ------------------------------------------------------------
    # Right panel: first-choice percentage
    # ------------------------------------------------------------
    bar_height = 0.16

    for group in observed_groups:
        summary = group_summaries[group]
        offset = group_offsets[group]
        color = group_colors[group]

        percentages = pd.to_numeric(
            summary["firstChoicePercent"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(percentages)
        y_values = positions + offset

        bars = ax_first.barh(
            y_values[valid],
            percentages[valid],
            height=bar_height,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.7,
            zorder=2,
        )

        for bar, percentage in zip(
                bars,
                percentages[valid],
        ):
            y_center = (
                    bar.get_y()
                    + bar.get_height() / 2
            )

            if percentage >= 14:
                x_position = percentage - 2
                horizontal_alignment = "right"
                text_color = "white"
            else:
                x_position = percentage + 2
                horizontal_alignment = "left"
                text_color = color

            ax_first.text(
                x_position,
                y_center,
                f"{percentage:.0f}%",
                ha=horizontal_alignment,
                va="center",
                fontsize=8,
                fontweight="semibold",
                color=text_color,
                zorder=3,
            )

    ax_first.set_xlim(0, 105)
    ax_first.set_xticks([0, 25, 50, 75, 100])
    ax_first.set_xlabel(
        "Participants ranking the workflow first (%)"
    )
    ax_first.set_title("First-choice share")

    # Y labels are already displayed in the left panel.
    ax_first.tick_params(
        axis="y",
        labelleft=False,
        length=0,
    )

    apply_standard_axes_style(
        ax_first,
        grid_axis="x",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=group_markers[group],
            linestyle="none",
            markersize=8,
            markerfacecolor=group_colors[group],
            markeredgecolor="white",
            label=(
                f"{group} "
                f"(n={group_sizes[group]})"
            ),
        )
        for group in observed_groups
    ]

    fig.legend(
        handles=legend_handles,
        title="Reported AI errors",
        bbox_to_anchor=(0.99, 0.5),
        loc="center left",
    )

    total_participants = ranking_df[
        "sessionId"
    ].nunique()

    fig.suptitle(
        (
            "Final Workflow Preference by "
            f"Reported AI Errors (N={total_participants})"
        ),
        fontsize=14,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Lower average ranks indicate stronger preference. The right panel "
            "shows the percentage of participants in each reported-error group "
            "who ranked a workflow first. Reported errors reflect participants' "
            "responses and do not necessarily indicate whether an AI error was "
            "actually encountered or correctly identified. Group comparisons "
            "are descriptive, particularly where group sizes are small."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color="#4A4A4A",
    )

    fig.tight_layout(
        rect=(0, 0.08, 0.82, 0.94)
    )

    save_figure(
        fig,
        slug,
        "Final Workflow Preference by Reported AI Errors",
        (
            "Average final workflow rank and first-choice share, separated by "
            "the number of AI errors participants reported. Lower mean ranks "
            "represent stronger preference. The full counts and percentages "
            "for every assigned rank are retained in the exported table."
        ),
    )


# ---------------------------------------------------------------------------
# 110: Interview awareness among exposed participants
# ---------------------------------------------------------------------------


def plot_injected_error_awareness(prepared) -> None:
    """Show whether exposed interview respondents noticed the injected error."""
    slug = "110_injected_error_awareness"

    notes = load_participant_interview_notes(prepared)
    required = {"injectedErrorExperience"}
    if notes.empty or not require_columns(
            notes, required, "injected-error awareness notes"
    ):
        return

    exposed = notes[notes["errorExposed"]].copy()
    if exposed.empty:
        print(
            "Skipping Figure 110; no exposed interview respondents with awareness coding were available."
        )
        return

    summary = (
        exposed["injectedErrorExperience"]
        .value_counts()
        .reindex(["noticed", "not_noticed"], fill_value=0)
        .rename_axis("awarenessCode")
        .reset_index(name="participantCount")
    )
    denominator = int(summary["participantCount"].sum())
    summary["awarenessLabel"] = summary["awarenessCode"].map(AWARENESS_LABELS)
    summary["percentage"] = summary["participantCount"] / denominator * 100
    summary["interviewRespondentsTotal"] = int(notes["participantId"].nunique())
    summary["exposedInterviewRespondents"] = denominator
    save_table(summary, slug, index=False)

    plot_df = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.barh(
        plot_df["awarenessLabel"],
        plot_df["percentage"],
        edgecolor=BAR_EDGE_COLOR,
    )
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['participantCount'])}/{denominator} ({row['percentage']:.1f}%)",
            va="center",
            fontsize=9,
            )
    ax.set_xlim(0, 112)
    ax.set_xlabel("Exposed interview respondents (%)")
    ax.set_ylabel("")
    ax.set_title("Awareness of the Injected AI Error")
    apply_standard_axes_style(ax, grid_axis="x")
    fig.text(
        0.01,
        0.01,
        f"Denominator: {denominator} interview respondent(s) confirmed as error-exposed in Main Round 1. Non-exposed respondents are excluded.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Awareness of the Injected AI Error",
        "Interview-coded awareness among respondents actually exposed to the injected error in Main Round 1.",
    )


# ---------------------------------------------------------------------------
# 111: Other AI error types reported in interviews
# ---------------------------------------------------------------------------


def plot_other_ai_error_types(prepared) -> None:
    """Show non-injected AI issues reported by interview respondents."""
    slug = "111_reported_other_ai_error_types"

    notes = load_participant_interview_notes(prepared)
    required = {"reportedOtherAiErrorTypes"}
    if notes.empty or not require_columns(
            notes, required, "other AI error interview notes"
    ):
        return

    total_respondents = int(notes["participantId"].nunique())
    rows = []
    for _, row in notes.iterrows():
        raw_types = row.get("reportedOtherAiErrorTypes")
        if pd.isna(raw_types) or not str(raw_types).strip():
            continue
        for raw_type in str(raw_types).split(";"):
            error_type = raw_type.strip()
            if error_type:
                rows.append(
                    {"participantId": row["participantId"], "errorType": error_type}
                )
    if not rows:
        print(
            "Skipping Figure 111; no coded non-injected AI error types were available."
        )
        return

    error_type_df = pd.DataFrame(rows).drop_duplicates(
        subset=["participantId", "errorType"]
    )

    unique_reporters = error_type_df["participantId"].nunique()
    total_issue_reports = len(error_type_df)

    summary = (
        pd.DataFrame(rows)
        .groupby("errorType")["participantId"]
        .nunique()
        .reset_index(name="participantCount")
    )
    summary["errorTypeLabel"] = (
        summary["errorType"]
        .map(OTHER_AI_ERROR_LABELS)
        .fillna(summary["errorType"].str.replace("_", " ").str.title())
    )
    summary["percentage"] = summary["participantCount"] / total_respondents * 100
    summary = summary.sort_values(
        ["participantCount", "errorTypeLabel"], ascending=[True, True]
    ).reset_index(drop=True)
    summary["interviewRespondentsTotal"] = total_respondents
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.4, max(4.5, 0.65 * len(summary) + 2.4)))
    bars = ax.barh(
        summary["errorTypeLabel"],
        summary["participantCount"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['participantCount'])} ({row['percentage']:.1f}%)",
            va="center",
            fontsize=8.5,
            )
    ax.set_xlim(0, max(1, summary["participantCount"].max() + 1.8))
    ax.set_xlabel("Interview respondents reporting the issue")
    ax.set_ylabel("")
    ax.set_title("Other AI Error Types Reported in Interviews")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    apply_standard_axes_style(ax, grid_axis="x")
    fig.text(
        0.01,
        0.01,
        (
            f"{unique_reporters} of {total_respondents} interview respondents "
            f"reported at least one other AI issue. "
            f"There were {total_issue_reports} issue-type reports in total; "
            "participants could report multiple issue types."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Other AI Error Types Reported in Interviews",
        "Interview-coded non-injected AI issues reported by participants. Categories are not mutually exclusive.",
    )


def plot_error_exposure(
        prepared: pd.DataFrame,
        feedback_df: pd.DataFrame,
) -> None:
    """Generate injected-error exposure and Main round outcome figures."""
    required = {
        "participantId",
        "roundIndex",
        "workflow",
        "errorExposed",
    }

    if prepared.empty or not require_columns(
            prepared,
            required,
            "error-exposure data",
    ):
        return

    prepared = add_passed_numeric(prepared)
    prepared["participantId"] = prepared["participantId"].astype(str)

    ranking_rows, _ = build_valid_ranking_rows(feedback_df)

    main_df = phase_data(prepared, "main")
    if main_df.empty:
        return

    # Exposure opportunity and manipulation check.
    plot_main_round1_workflow_choice(prepared)

    # Participant experience during voluntary Main rounds.
    plot_main_round_satisfaction_by_exposure(main_df)
    plot_main_round_ai_experience_by_exposure(main_df)
    plot_main_round_tlx_by_exposure_and_workflow(main_df)

    # Post-error behavior and output outcomes.
    plot_post_error_workflow_choices_by_exposure(prepared)
    plot_main_round_quality_by_error_exposure(main_df)
    plot_main_constraint_fulfillment_by_exposure(main_df)
    plot_main_line_count_error_by_exposure(main_df)

    # Final preferences and interview-coded reflections.
    plot_final_workflow_preference_by_reported_ai_errors(
        ranking_rows,
        prepared,
    )
    plot_injected_error_awareness(prepared)
    plot_other_ai_error_types(prepared)
