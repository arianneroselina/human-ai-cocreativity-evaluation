"""Output-quality figures for the research dashboard.

Figures
--------------------
11  Mean overall quality by workflow in main rounds
12  Rating dimensions by workflow in main rounds
13  Mean overall quality versus completion time by workflow
14  Human→AI versus AI→Human quality in main rounds
15  Mixed versus solo workflow quality in main rounds
16  Main-round quality trends by actual error-exposure group
17  Highest- and lowest-rated poems
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scripts.config import (
    MAIN_ROUND_INDICES,
    QUALITY_DIMENSION_LABELS,
    QUALITY_PRIMARY_METRIC,
    QUALITY_SCALE_MAX,
    QUALITY_SCALE_MIN,
    WORKFLOW_COLORS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    workflow_display_name,
    exposure_display_name,
    drop_duplicate_participant_rounds,
    phase_data,
    ordered_exposure_groups,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


QUALITY_Y_MIN = QUALITY_SCALE_MIN - 0.5
QUALITY_Y_MAX = QUALITY_SCALE_MAX + 0.5
CI_Z_VALUE = 1.96


# -----------------------------------------------------------------------------
# Shared quality-data helpers
# -----------------------------------------------------------------------------


def _prepare_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid observations for the primary quality outcome."""
    prepared = df.copy()

    if QUALITY_PRIMARY_METRIC not in prepared.columns:
        return pd.DataFrame()

    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    return prepared.dropna(subset=[QUALITY_PRIMARY_METRIC]).copy()


def _quality_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
) -> pd.DataFrame:
    """Calculate descriptive mean, spread, and normal-approximation 95% CI."""
    summary = (
        dataframe.groupby(group_columns, dropna=False)[metric]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ciLow"] = summary["mean"] - CI_Z_VALUE * summary["se"]
    summary["ciHigh"] = summary["mean"] + CI_Z_VALUE * summary["se"]
    summary.loc[summary["count"] < 2, ["std", "se", "ciLow", "ciHigh"]] = np.nan

    return summary


def _workflow_order_present(dataframe: pd.DataFrame) -> list[str]:
    """Keep canonical workflow order while omitting unavailable categories."""
    available = set(dataframe["workflow"].dropna().unique())
    return [workflow for workflow in WORKFLOW_ORDER if workflow in available]


def _add_raw_points(
    ax,
    dataframe: pd.DataFrame,
    workflows: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
    seed: int = 42,
) -> None:
    """Add reproducibly jittered poem-level observations over a category plot."""
    rng = np.random.default_rng(seed)

    for position, workflow in enumerate(workflows, start=1):
        values = (
            dataframe.loc[
                dataframe["workflow"] == workflow,
                metric,
            ]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=WORKFLOW_COLORS[workflow],
            alpha=0.48,
            s=28,
            zorder=3,
            linewidths=0,
        )


def _add_mean_intervals(
    ax,
    summary: pd.DataFrame,
    workflows: list[str],
) -> None:
    """Overlay workflow means and descriptive 95% confidence intervals."""
    for position, workflow in enumerate(workflows, start=1):
        row = summary.loc[summary["workflow"] == workflow]
        if row.empty:
            continue

        mean = float(row.iloc[0]["mean"])
        low = row.iloc[0]["ciLow"]
        high = row.iloc[0]["ciHigh"]

        if pd.notna(low) and pd.notna(high):
            yerr = [[mean - low], [high - mean]]
            ax.errorbar(
                position,
                mean,
                yerr=yerr,
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


def _annotate_workflow_counts(
    ax,
    summary: pd.DataFrame,
    workflows: list[str],
    y_position: float = QUALITY_Y_MIN + 0.12,
) -> None:
    """Annotate category plots with the number of quality observations."""
    for position, workflow in enumerate(workflows, start=1):
        row = summary.loc[summary["workflow"] == workflow]
        if row.empty:
            continue
        ax.text(
            position,
            y_position,
            f"n={int(row.iloc[0]['count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_workflow_quality_distribution(
    dataframe: pd.DataFrame,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Create a raw-point + boxplot + mean/CI quality comparison."""
    if dataframe.empty:
        return

    workflows = _workflow_order_present(dataframe)
    if not workflows:
        return

    summary = _quality_summary(dataframe, ["workflow"])
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    box_data = [
        dataframe.loc[dataframe["workflow"] == workflow, QUALITY_PRIMARY_METRIC]
        .dropna()
        .to_numpy()
        for workflow in workflows
    ]

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

    _add_raw_points(ax, dataframe, workflows)
    _add_mean_intervals(ax, summary, workflows)
    _annotate_workflow_counts(ax, summary, workflows)

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
    ax.set_title(title)
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean overall quality (1-5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.tick_params(axis="x", rotation=0)
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)


def _paired_matrix(
    dataframe: pd.DataFrame,
    workflows: list[str],
) -> pd.DataFrame:
    """Build a participant-by-workflow matrix for a specified workflow set."""
    required = {"participantId", "workflow", QUALITY_PRIMARY_METRIC}
    if not require_columns(dataframe, required, "paired quality comparison"):
        return pd.DataFrame()

    matrix = (
        dataframe.loc[dataframe["workflow"].isin(workflows)]
        .pivot_table(
            index="participantId",
            columns="workflow",
            values=QUALITY_PRIMARY_METRIC,
            aggfunc="first",
        )
        .reindex(columns=workflows)
    )

    return matrix.dropna(how="any")


def _plot_two_condition_paired_comparison(
    paired_df: pd.DataFrame,
    left_column: str,
    right_column: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Draw a within-participant two-condition quality comparison."""
    if paired_df.empty:
        return

    export_df = paired_df.reset_index(drop=True).copy()
    export_df.insert(
        0,
        "participantCode",
        [f"P{index:02d}" for index in range(1, len(export_df) + 1)],
    )
    export_df = export_df.rename(
        columns={left_column: left_label, right_column: right_label}
    )
    save_table(export_df, slug, index=False)

    values = paired_df[[left_column, right_column]]
    summary = _quality_summary(
        values.melt(var_name="comparison", value_name=QUALITY_PRIMARY_METRIC),
        ["comparison"],
    )
    save_table(summary, f"{slug}_summary", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    rng = np.random.default_rng(42)
    x_positions = np.array([1.0, 2.0])

    for _, row in paired_df.iterrows():
        jitter = rng.uniform(-0.035, 0.035, size=2)
        x_values = x_positions + jitter
        y_values = [row[left_column], row[right_column]]
        ax.plot(x_values, y_values, color="0.65", alpha=0.55, linewidth=1.0, zorder=1)
        ax.scatter(
            x_values[0], y_values[0], color=left_color, alpha=0.75, s=30, zorder=2
        )
        ax.scatter(
            x_values[1], y_values[1], color=right_color, alpha=0.75, s=30, zorder=2
        )

    for x_position, comparison, color in zip(
        x_positions,
        [left_column, right_column],
        [left_color, right_color],
    ):
        row = summary.loc[summary["comparison"] == comparison].iloc[0]
        mean = float(row["mean"])
        low = row["ciLow"]
        high = row["ciHigh"]

        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                x_position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color="black",
                markerfacecolor=color,
                markeredgecolor="black",
                capsize=4,
                markersize=7,
                zorder=5,
            )
        else:
            ax.scatter(
                x_position,
                mean,
                marker="D",
                color="black",
                facecolor=color,
                s=55,
                zorder=5,
            )

    ax.text(
        0.5,
        0.03,
        f"Paired participants: n={len(paired_df)}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([left_label, right_label])
    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.set_title(title)
    ax.set_ylabel("Mean overall quality (1-5)")
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)


# -----------------------------------------------------------------------------
# 11: Main rounds workflow quality
# -----------------------------------------------------------------------------


def plot_mean_overall_quality_by_workflow_main_rounds(main_df) -> None:
    """Compare quality across voluntarily selected workflows in Main rounds."""
    _plot_workflow_quality_distribution(
        main_df,
        slug="11_mean_overall_quality_by_workflow_main_rounds",
        title="Mean Overall Quality by Workflow in Main Rounds",
        description=(
            "Poem-level mean overall quality in the Main rounds. "
            "Participants selected their own workflow, so differences are "
            "descriptive associations rather than controlled workflow effects."
        ),
    )


# def plot_mean_overall_quality_by_workflow_practice_rounds(df: pd.DataFrame) -> None:
#     """Compare quality in the controlled, assigned-workflow practice phase."""
#     main_df = phase_data(_prepare_quality_data(df), "practice")
#     _plot_workflow_quality_distribution(
#         main_df,
#         slug="11b_mean_overall_quality_by_workflow_practice_rounds",
#         title="Mean Overall Quality by Workflow in Practice Rounds",
#         description="Poem-level mean overall quality in the assigned-workflow practice "
#         "phase. Workflow and task order were randomized, making this the primary "
#         "descriptive workflow-quality comparison.",
#     )


# -----------------------------------------------------------------------------
# 12: Main rounds dimension-level quality
# -----------------------------------------------------------------------------


def plot_rating_dimensions_by_workflow_main_rounds(main_df) -> None:
    """Compare evaluator-rated quality dimensions in main rounds."""
    slug = "12_rating_dimensions_by_workflow_main_rounds"

    available_dimensions = [
        column
        for column in QUALITY_DIMENSION_LABELS
        if column in main_df.columns
        and pd.to_numeric(main_df[column], errors="coerce").notna().any()
    ]
    if not available_dimensions:
        print(
            "Skipping quality dimensions; no configured dimension columns are available."
        )
        return

    long_rows = []
    for column in available_dimensions:
        values = pd.to_numeric(main_df[column], errors="coerce")
        dimension_df = main_df.loc[values.notna(), ["workflow"]].copy()
        dimension_df["dimension"] = QUALITY_DIMENSION_LABELS[column]
        dimension_df["score"] = values.loc[values.notna()].to_numpy()
        long_rows.append(dimension_df)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df = long_df[long_df["workflow"].isin(WORKFLOW_ORDER)]
    if long_df.empty:
        return

    summary = _quality_summary(
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
            summary[summary["workflow"] == workflow]
            .set_index("dimension")
            .reindex(dimension_labels)
        )
        y_positions = base_positions + offset
        means = workflow_summary["mean"].to_numpy()
        lows = workflow_summary["ciLow"].to_numpy()
        highs = workflow_summary["ciHigh"].to_numpy()

        lower_errors = np.where(pd.notna(lows), means - lows, 0.0)
        upper_errors = np.where(pd.notna(highs), highs - means, 0.0)
        ax.errorbar(
            means,
            y_positions,
            xerr=np.vstack([lower_errors, upper_errors]),
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
    ax.set_title("Quality Dimensions by Workflow in Main Rounds")
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="x")

    save_figure(
        fig,
        slug,
        "Quality Dimensions by Workflow in Main Rounds",
        "Mean evaluator ratings by workflow and quality dimension in main rounds "
        "rounds. Error bars show descriptive 95% confidence intervals.",
    )


# -----------------------------------------------------------------------------
# 13: Quality–time relationship
# -----------------------------------------------------------------------------


def plot_quality_vs_completion_time_by_workflow(prepared) -> None:
    """Show poem-level mean overall quality and completion time without ratio-score emphasis."""
    slug = "13_quality_vs_completion_time_by_workflow"

    if "effectiveTimeMinutes" in prepared.columns:
        prepared["completionTimeMinutes"] = pd.to_numeric(
            prepared["effectiveTimeMinutes"],
            errors="coerce",
        )
    elif "timeMs" in prepared.columns:
        prepared["completionTimeMinutes"] = (
            pd.to_numeric(
                prepared["timeMs"],
                errors="coerce",
            )
            / 60000
        )
    else:
        print("Skipping quality-time plot; missing effectiveTimeMinutes and timeMs.")
        return

    main_df = prepared.dropna(subset=["completionTimeMinutes"])
    main_df = main_df[main_df["completionTimeMinutes"] >= 0].copy()
    if main_df.empty:
        return

    summary = (
        main_df.groupby("workflow")
        .agg(
            meanQuality=(QUALITY_PRIMARY_METRIC, "mean"),
            meanCompletionTimeMinutes=("completionTimeMinutes", "mean"),
            count=(QUALITY_PRIMARY_METRIC, "count"),
        )
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    workflows = _workflow_order_present(main_df)

    for workflow in workflows:
        workflow_df = main_df[main_df["workflow"] == workflow]
        ax.scatter(
            workflow_df["completionTimeMinutes"],
            workflow_df[QUALITY_PRIMARY_METRIC],
            color=WORKFLOW_COLORS[workflow],
            alpha=0.52,
            s=34,
            label=workflow_display_name(workflow),
        )

        workflow_summary = summary.loc[summary["workflow"] == workflow].iloc[0]
        mean_x = workflow_summary["meanCompletionTimeMinutes"]
        mean_y = workflow_summary["meanQuality"]
        ax.scatter(
            mean_x,
            mean_y,
            marker="D",
            s=82,
            color=WORKFLOW_COLORS[workflow],
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            f"{workflow_display_name(workflow)}\nmean n={int(workflow_summary['count'])}",
            xy=(mean_x, mean_y),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Mean Overall Quality and Completion Time by Workflow")
    ax.set_xlabel("Effective completion time (minutes)")
    ax.set_ylabel("Mean Overall quality (1-5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Mean Overall Quality and Completion Time by Workflow",
        "Each point is one evaluated output. Larger diamonds show the mean time "
        "and mean quality for each workflow; the plot displays the quality-time "
        "trade-off directly rather than using a ratio score.",
    )


# -----------------------------------------------------------------------------
# 14: Paired workflows quality in main rounds
# -----------------------------------------------------------------------------


def plot_mixed_workflow_direction_quality_main_rounds(main_df) -> None:
    """Compare each participant's Human→AI and AI→Human workflow quality in main rounds."""
    slug = "14_mixed_workflow_direction_quality_main_rounds"
    paired_df = _paired_matrix(main_df, ["human_ai", "ai_human"])

    _plot_two_condition_paired_comparison(
        paired_df=paired_df,
        left_column="human_ai",
        right_column="ai_human",
        left_label="Human → AI",
        right_label="AI → Human",
        left_color=WORKFLOW_COLORS["human_ai"],
        right_color=WORKFLOW_COLORS["ai_human"],
        slug=slug,
        title="Mixed-Workflow Direction and Quality in Main Rounds",
        description="Within-participant comparison of the two mixed workflows in "
        "main rounds. Lines connect the same participant across "
        "Human→AI and AI→Human.",
    )


def plot_mixed_vs_solo_quality_main_rounds(main_df) -> None:
    """Compare each participant's mixed- and solo-workflow quality in main rounds."""
    slug = "15_mixed_vs_solo_quality_main_rounds"

    required_workflows = ["human", "ai", "human_ai", "ai_human"]
    matrix = _paired_matrix(main_df, required_workflows)
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
        paired_df=paired_df,
        left_column="solo",
        right_column="mixed",
        left_label="Solo workflows\n(Human-only + AI-only)",
        right_label="Mixed workflows\n(Human→AI + AI→Human)",
        left_color="0.45",
        right_color=WORKFLOW_COLORS["human_ai"],
        slug=slug,
        title="Mixed versus Solo Workflow Quality in Main Rounds",
        description="Within-participant comparison of average quality in solo and "
        "mixed workflows in main rounds. Each participant contributes one average over Human-only "
        "and AI-only outputs and one average over Human→AI and AI→Human outputs.",
    )


# -----------------------------------------------------------------------------
# 16: Main-round quality patterns by actual error exposure
# -----------------------------------------------------------------------------


def plot_main_round_quality_by_error_exposure(main_df) -> None:
    """Plot descriptive Main-round quality trends by AI-error exposure."""
    slug = "16_main_round_quality_by_error_exposure"

    if not require_columns(
        main_df,
        {"roundIndex", "errorExposed"},
        "main-round quality by exposure",
    ):
        return

    main_df["roundIndex"] = pd.to_numeric(
        main_df["roundIndex"],
        errors="coerce",
    )

    main_df = main_df.dropna(subset=["roundIndex", "errorExposed"])
    main_df["roundIndex"] = main_df["roundIndex"].astype(int)
    main_df = main_df[main_df["roundIndex"].isin(MAIN_ROUND_INDICES)].copy()

    groups = ordered_exposure_groups(main_df)

    if not groups:
        return

    summary = _quality_summary(
        main_df,
        ["errorExposed", "roundIndex", "workflow"],
    )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)

    save_table(summary, slug, index=False)

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.6 * len(groups), 5.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for panel_index, exposed in enumerate(groups):
        ax = axes[panel_index]

        group_summary = summary[summary["errorExposed"].eq(exposed)]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = group_summary[
                group_summary["workflow"].eq(workflow)
            ].sort_values("roundIndex")

            if workflow_summary.empty:
                continue

            rounds = workflow_summary["roundIndex"].to_numpy()
            means = workflow_summary["mean"].to_numpy()
            lows = workflow_summary["ciLow"].to_numpy()
            highs = workflow_summary["ciHigh"].to_numpy()

            ax.plot(
                rounds,
                means,
                marker="o",
                linewidth=2,
                markersize=5,
                color=WORKFLOW_COLORS[workflow],
                label=workflow_display_name(workflow),
            )

            valid_ci = pd.notna(lows) & pd.notna(highs)

            if valid_ci.any():
                ax.fill_between(
                    rounds[valid_ci],
                    lows[valid_ci],
                    highs[valid_ci],
                    color=WORKFLOW_COLORS[workflow],
                    alpha=0.14,
                )

        ax.set_title(exposure_display_name(exposed))
        ax.set_xticks(MAIN_ROUND_INDICES)
        ax.set_xticklabels(
            [f"Main {index}" for index in range(1, len(MAIN_ROUND_INDICES) + 1)]
        )
        ax.set_xlabel("Free-choice main round")
        ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)

        if panel_index == 0:
            ax.set_ylabel("Mean overall quality (1-5)")

        apply_standard_axes_style(ax)

    axes[-1].legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.suptitle(
        "Mean Overall Quality Across Main Rounds by Error Exposure",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(
        fig,
        slug,
        "Mean Overall Quality Across Main Rounds by Error Exposure",
        "Descriptive workflow-level quality trends in Main rounds, shown "
        "separately for participants exposed and not exposed to the injected "
        "AI error in round 5.",
    )


# -----------------------------------------------------------------------------
# 17: Highest- and lowest-rated poems
# -----------------------------------------------------------------------------


def plot_quality_examples(prepared, max_examples_per_extreme=3):
    """Create optional illustrative examples of highest- and lowest-rated poems.

    These figures are for curious dashboard readers only and are not intended
    as evidence of a general workflow effect.
    """
    required_columns = {QUALITY_PRIMARY_METRIC, "text", "workflow", "roundIndex"}

    if not require_columns(prepared, required_columns, "quality examples"):
        return

    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC], errors="coerce"
    )
    prepared = prepared.dropna(
        subset=[QUALITY_PRIMARY_METRIC, "text", "workflow", "roundIndex"]
    )

    if prepared.empty:
        return

    prepared["text"] = prepared["text"].astype(str).str.strip()
    prepared = prepared[prepared["text"] != ""]

    if prepared.empty:
        return

    display_columns = [
        "roundId",
        "roundIndex",
        "workflow",
        "topic",
        "text",
        QUALITY_PRIMARY_METRIC,
        "constraintScore",
        "ratingCount",
    ]
    export_columns = [
        column for column in display_columns if column in prepared.columns
    ]

    def wrap_text(text, max_line_length=72):
        wrapped_lines = []

        for original_line in str(text).splitlines():
            line = original_line.strip()

            if not line:
                wrapped_lines.append("")
                continue

            while len(line) > max_line_length:
                split_index = line.rfind(" ", 0, max_line_length)

                if split_index == -1:
                    split_index = max_line_length

                wrapped_lines.append(line[:split_index].strip())
                line = line[split_index:].strip()

            wrapped_lines.append(line)

        return "\n".join(wrapped_lines)

    def select_examples(dataframe, ascending=False):
        """Select up to max_examples_per_extreme, preserving score order."""
        sorted_df = dataframe.sort_values(
            by=[QUALITY_PRIMARY_METRIC, "roundIndex"],
            ascending=[ascending, True],
            kind="stable",
        )

        return sorted_df.head(max_examples_per_extreme).copy()

    def render_examples(example_df, slug, title, description):
        if example_df.empty:
            return

        example_df = example_df.copy()
        example_df["workflowLabel"] = (
            example_df["workflow"].map(WORKFLOW_LABELS).fillna(example_df["workflow"])
        )

        # Conservative spacing values in inches.
        HEADER_LINE_HEIGHT = 0.21
        TOPIC_LINE_HEIGHT = 0.18
        POEM_LINE_HEIGHT = 0.21

        HEADER_TO_TOPIC_GAP = 0.08
        TOPIC_TO_POEM_GAP = 0.12
        POEM_BOX_PADDING = 0.22
        BLOCK_BOTTOM_GAP = 0.38

        TOP_PADDING = 0.85
        BOTTOM_PADDING = 0.30

        def count_lines(value):
            return str(value).count("\n") + 1

        examples = []

        for index, (_, row) in enumerate(example_df.iterrows(), start=1):
            topic = (
                row.get("topic") if pd.notna(row.get("topic")) else "Topic unavailable"
            )

            metadata = [
                f"Round {int(row['roundIndex'])}",
                str(row["workflowLabel"]),
                f"Quality: {row[QUALITY_PRIMARY_METRIC]:.2f}/5",
            ]

            if "constraintScore" in row and pd.notna(row["constraintScore"]):
                metadata.append(
                    f"Constraint score: {float(row['constraintScore']):.2f}"
                )

            if "ratingCount" in row and pd.notna(row["ratingCount"]):
                metadata.append(f"Ratings: {int(row['ratingCount'])}")

            header = f"Example {index}  |  " + "  |  ".join(metadata)

            # Wrap these too, in case metadata or topics become long.
            header = wrap_text(header, max_line_length=135)
            topic_text = wrap_text(f"Topic: {topic}", max_line_length=110)
            poem_text = wrap_text(row["text"])

            block_height = (
                count_lines(header) * HEADER_LINE_HEIGHT
                + HEADER_TO_TOPIC_GAP
                + count_lines(topic_text) * TOPIC_LINE_HEIGHT
                + TOPIC_TO_POEM_GAP
                + count_lines(poem_text) * POEM_LINE_HEIGHT
                + POEM_BOX_PADDING
                + BLOCK_BOTTOM_GAP
            )

            examples.append(
                {
                    "header": header,
                    "topic": topic_text,
                    "poem": poem_text,
                    "height": block_height,
                }
            )

        figure_height = max(
            5.0,
            TOP_PADDING
            + BOTTOM_PADDING
            + sum(example["height"] for example in examples),
        )

        fig = plt.figure(figsize=(11.5, figure_height))
        fig.patch.set_facecolor("white")

        fig.text(
            0.5,
            (figure_height - 0.25) / figure_height,
            title,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

        y_position = figure_height - TOP_PADDING

        for example in examples:
            fig.text(
                0.03,
                y_position / figure_height,
                example["header"],
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

            y_position -= count_lines(example["header"]) * HEADER_LINE_HEIGHT
            y_position -= HEADER_TO_TOPIC_GAP

            fig.text(
                0.03,
                y_position / figure_height,
                example["topic"],
                ha="left",
                va="top",
                fontsize=9,
                style="italic",
            )

            y_position -= count_lines(example["topic"]) * TOPIC_LINE_HEIGHT
            y_position -= TOPIC_TO_POEM_GAP

            fig.text(
                0.03,
                y_position / figure_height,
                example["poem"],
                ha="left",
                va="top",
                fontsize=9,
                family="monospace",
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.55",
                    "facecolor": "#ffffff",
                    "edgecolor": "#d9d9d9",
                    "linewidth": 1,
                },
            )

            y_position -= count_lines(example["poem"]) * POEM_LINE_HEIGHT
            y_position -= POEM_BOX_PADDING
            y_position -= BLOCK_BOTTOM_GAP

        save_figure(fig, slug, title, description)

    highest_examples = select_examples(prepared, ascending=False)
    lowest_examples = select_examples(prepared, ascending=True)

    highest_slug = "17_highest_rated_poem_examples"
    lowest_slug = "17b_lowest_rated_poem_examples"

    save_table(
        highest_examples[export_columns],
        highest_slug,
        index=False,
    )
    save_table(
        lowest_examples[export_columns],
        lowest_slug,
        index=False,
    )

    render_examples(
        highest_examples,
        highest_slug,
        "Highest-Rated Poem Examples",
        "Illustrative examples selected by the highest mean overall quality scores. "
        "They are individual cases and should not be interpreted as evidence "
        "that a workflow generally performs better.",
    )

    render_examples(
        lowest_examples,
        lowest_slug,
        "Lowest-Rated Poem Examples",
        "Illustrative examples selected by the lowest mean overall quality scores. "
        "They are individual cases and should not be interpreted as evidence "
        "that a workflow generally performs worse.",
    )


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------


def plot_quality(df: pd.DataFrame) -> None:
    """Generate the complete output-quality figure set."""
    prepared = _prepare_quality_data(df)
    if prepared.empty:
        return

    main_df = phase_data(prepared, "main")
    if main_df.empty:
        return

    plot_mean_overall_quality_by_workflow_main_rounds(main_df)
    plot_rating_dimensions_by_workflow_main_rounds(main_df)
    plot_quality_vs_completion_time_by_workflow(prepared)
    plot_mixed_workflow_direction_quality_main_rounds(main_df)
    plot_mixed_vs_solo_quality_main_rounds(main_df)
    plot_main_round_quality_by_error_exposure(main_df)
    plot_quality_examples(main_df)
