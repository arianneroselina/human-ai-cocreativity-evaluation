"""Output-quality figures for the research dashboard.

Core quality figures
--------------------
11  Evaluation coverage across submitted outputs
12  Composite quality by workflow across all evaluated rounds
13  Composite quality by workflow in controlled practice rounds
14  Quality dimensions by workflow in controlled practice rounds
15  Composite quality versus completion time by workflow
16  Human→AI versus AI→Human quality in practice rounds
17  Mixed versus solo workflow quality in practice rounds
18  Main-round quality trends by actual error-exposure group
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scripts.config import (
    EXPECTED_EVALUATORS,
    EXPOSURE_LABELS,
    MAIN_ROUND_INDICES,
    PRACTICE_ROUND_INDICES,
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
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    drop_duplicate_participant_rounds,
    parse_bool,
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


def _truthy_mask(series: pd.Series) -> pd.Series:
    """Interpret common CSV boolean representations as a boolean mask."""
    normalized = series.astype(str).str.strip().str.lower()
    return series.map(parse_bool) | normalized.isin({"1.0"})


def _prepare_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return deduplicated rows with valid primary-quality observations."""
    required = {QUALITY_PRIMARY_METRIC, "workflow"}
    if df.empty or not require_columns(df, required, "quality data"):
        return pd.DataFrame()

    prepared = drop_duplicate_participant_rounds(df.copy())
    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=[QUALITY_PRIMARY_METRIC, "workflow"])
    prepared = prepared[prepared["workflow"].isin(WORKFLOW_ORDER)].copy()

    return prepared


def _phase_data(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Filter rows to the planned practice or main phase without guessing values."""
    if phase not in {"practice", "main"}:
        raise ValueError("phase must be either 'practice' or 'main'")

    flag_column = "isPracticeRound" if phase == "practice" else "isMainRound"
    round_indices = (
        PRACTICE_ROUND_INDICES if phase == "practice" else MAIN_ROUND_INDICES
    )

    if flag_column in df.columns:
        return df.loc[_truthy_mask(df[flag_column])].copy()

    if "roundIndex" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    numeric_rounds = pd.to_numeric(df["roundIndex"], errors="coerce")
    return df.loc[numeric_rounds.isin(round_indices)].copy()


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


def _fully_rated_mask(dataframe: pd.DataFrame) -> pd.Series:
    """Identify fully rated outputs from the explicit flag or rating count."""
    if "isFullyRated" in dataframe.columns:
        return _truthy_mask(dataframe["isFullyRated"])

    if "ratingCount" in dataframe.columns:
        rating_count = pd.to_numeric(dataframe["ratingCount"], errors="coerce")
        return rating_count.ge(EXPECTED_EVALUATORS)

    return pd.Series(False, index=dataframe.index)


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
    ax.set_ylabel("Composite quality (1–5)")
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
    ax.set_ylabel("Composite quality (1–5)")
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)


# -----------------------------------------------------------------------------
# 11: Evaluation coverage
# -----------------------------------------------------------------------------


def plot_evaluation_coverage(df: pd.DataFrame) -> None:
    """Show how many outputs received ratings and reached full rating coverage."""
    slug = "11_evaluation_coverage"
    if df.empty:
        return

    coverage_df = drop_duplicate_participant_rounds(df.copy())
    if coverage_df.empty:
        return

    has_rating_count = "ratingCount" in coverage_df.columns
    has_full_flag = "isFullyRated" in coverage_df.columns
    if not has_rating_count and not has_full_flag:
        print("Skipping evaluation coverage; missing ratingCount and isFullyRated.")
        return

    if has_rating_count:
        coverage_df["ratingCountNumeric"] = (
            pd.to_numeric(
                coverage_df["ratingCount"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        coverage_df["ratingCountNumeric"] = np.nan

    coverage_df["isFullyRatedCalculated"] = _fully_rated_mask(coverage_df)

    rating_distribution = (
        coverage_df["ratingCountNumeric"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("ratingCount")
        .reset_index(name="outputCount")
    )
    save_table(rating_distribution, f"{slug}_rating_count_distribution", index=False)

    if "workflow" in coverage_df.columns:
        workflow_coverage = (
            coverage_df[coverage_df["workflow"].isin(WORKFLOW_ORDER)]
            .groupby("workflow")
            .agg(
                outputCount=("workflow", "size"),
                outputsWithAtLeastOneRating=(
                    "ratingCountNumeric",
                    lambda values: (
                        int((values > 0).sum()) if has_rating_count else np.nan
                    ),
                ),
                fullyRated=("isFullyRatedCalculated", "sum"),
            )
            .reindex(WORKFLOW_ORDER, fill_value=0)
            .reset_index()
        )
        workflow_coverage["fullyRatedPercentage"] = np.where(
            workflow_coverage["outputCount"] > 0,
            workflow_coverage["fullyRated"] / workflow_coverage["outputCount"] * 100,
            np.nan,
        )
        workflow_coverage["workflowLabel"] = workflow_coverage["workflow"].map(
            workflow_display_name
        )
        save_table(workflow_coverage, f"{slug}_by_workflow", index=False)
    else:
        workflow_coverage = pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    if has_rating_count and not rating_distribution.empty:
        bars = axes[0].bar(
            rating_distribution["ratingCount"].astype(str),
            rating_distribution["outputCount"],
            edgecolor=BAR_EDGE_COLOR,
        )
        for bar, count in zip(bars, rating_distribution["outputCount"]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(len(coverage_df) * 0.01, 0.1),
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[0].set_title("Number of Evaluator Ratings per Output")
        axes[0].set_xlabel("Rating count")
        axes[0].set_ylabel("Outputs")
        apply_standard_axes_style(axes[0])
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "Rating-count data unavailable",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    if not workflow_coverage.empty:
        bars = axes[1].bar(
            workflow_coverage["workflowLabel"],
            workflow_coverage["fullyRatedPercentage"],
            color=[
                WORKFLOW_COLORS[workflow] for workflow in workflow_coverage["workflow"]
            ],
            edgecolor=BAR_EDGE_COLOR,
        )
        for bar, (_, row) in zip(bars, workflow_coverage.iterrows()):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{int(row['fullyRated'])}/{int(row['outputCount'])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axes[1].set_title("Fully Rated Outputs by Workflow")
        axes[1].set_xlabel("Workflow")
        axes[1].set_ylabel("Fully rated outputs (%)")
        axes[1].set_ylim(0, 112)
        axes[1].tick_params(axis="x", rotation=12)
        apply_standard_axes_style(axes[1])
    else:
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Workflow data unavailable",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    fig.tight_layout()
    save_figure(
        fig,
        slug,
        "Evaluation Coverage",
        "Number of evaluator ratings per submitted output and the share of outputs "
        "that received all expected evaluator ratings, shown by workflow.",
    )


# -----------------------------------------------------------------------------
# 12–13: Overall and controlled-practice workflow quality
# -----------------------------------------------------------------------------


def plot_composite_quality_by_workflow_all_rounds(df: pd.DataFrame) -> None:
    """Compare externally rated composite quality across all evaluated rounds."""
    plot_df = _prepare_quality_data(df)
    _plot_workflow_quality_distribution(
        plot_df,
        slug="12_composite_quality_by_workflow_all_rounds",
        title="Composite Quality by Workflow Across All Evaluated Rounds",
        description="Poem-level composite quality across all rounds with a valid "
        "external quality score. Points are individual outputs; diamonds show "
        "workflow means with descriptive 95% confidence intervals.",
    )


def plot_composite_quality_by_workflow_practice_rounds(df: pd.DataFrame) -> None:
    """Compare quality in the controlled, assigned-workflow practice phase."""
    plot_df = _phase_data(_prepare_quality_data(df), "practice")
    _plot_workflow_quality_distribution(
        plot_df,
        slug="13_composite_quality_by_workflow_practice_rounds",
        title="Composite Quality by Workflow in Controlled Practice Rounds",
        description="Poem-level composite quality in the assigned-workflow practice "
        "phase. Workflow and task order were randomized, making this the primary "
        "descriptive workflow-quality comparison.",
    )


# -----------------------------------------------------------------------------
# 14: Dimension-level quality profile
# -----------------------------------------------------------------------------


def plot_quality_dimensions_by_workflow_practice_rounds(df: pd.DataFrame) -> None:
    """Compare evaluator-rated quality dimensions in controlled practice rounds."""
    slug = "14_quality_dimensions_by_workflow_practice_rounds"
    plot_df = _phase_data(_prepare_quality_data(df), "practice")
    if plot_df.empty:
        return

    available_dimensions = [
        column
        for column in QUALITY_DIMENSION_LABELS
        if column in plot_df.columns
        and pd.to_numeric(plot_df[column], errors="coerce").notna().any()
    ]
    if not available_dimensions:
        print(
            "Skipping quality dimensions; no configured dimension columns are available."
        )
        return

    long_rows = []
    for column in available_dimensions:
        values = pd.to_numeric(plot_df[column], errors="coerce")
        dimension_df = plot_df.loc[values.notna(), ["workflow"]].copy()
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
    ax.set_xlabel("Mean evaluator rating (1–5)")
    ax.set_ylabel("Quality dimension")
    ax.set_title("Quality Dimensions by Workflow in Controlled Practice Rounds")
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="x")

    save_figure(
        fig,
        slug,
        "Quality Dimensions by Workflow in Controlled Practice Rounds",
        "Mean evaluator ratings by workflow and quality dimension in practice "
        "rounds. Error bars show descriptive 95% confidence intervals.",
    )


# -----------------------------------------------------------------------------
# 15: Quality–time relationship
# -----------------------------------------------------------------------------


def plot_quality_vs_completion_time_by_workflow(df: pd.DataFrame) -> None:
    """Show poem-level quality and completion time without ratio-score emphasis."""
    slug = "15_quality_vs_completion_time_by_workflow"
    plot_df = _prepare_quality_data(df)
    if plot_df.empty:
        return

    if "effectiveTimeMinutes" in plot_df.columns:
        plot_df["completionTimeMinutes"] = pd.to_numeric(
            plot_df["effectiveTimeMinutes"],
            errors="coerce",
        )
    elif "timeMs" in plot_df.columns:
        plot_df["completionTimeMinutes"] = (
            pd.to_numeric(
                plot_df["timeMs"],
                errors="coerce",
            )
            / 60000
        )
    else:
        print("Skipping quality-time plot; missing effectiveTimeMinutes and timeMs.")
        return

    plot_df = plot_df.dropna(subset=["completionTimeMinutes"])
    plot_df = plot_df[plot_df["completionTimeMinutes"] >= 0].copy()
    if plot_df.empty:
        return

    summary = (
        plot_df.groupby("workflow")
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
    workflows = _workflow_order_present(plot_df)

    for workflow in workflows:
        workflow_df = plot_df[plot_df["workflow"] == workflow]
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

    ax.set_title("Composite Quality and Completion Time by Workflow")
    ax.set_xlabel("Effective completion time (minutes)")
    ax.set_ylabel("Composite quality (1–5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Composite Quality and Completion Time by Workflow",
        "Each point is one evaluated output. Larger diamonds show the mean time "
        "and mean quality for each workflow; the plot displays the quality-time "
        "trade-off directly rather than using a ratio score.",
    )


# -----------------------------------------------------------------------------
# 16–17: Paired controlled-practice comparisons
# -----------------------------------------------------------------------------


def plot_mixed_workflow_direction_quality_practice_rounds(df: pd.DataFrame) -> None:
    """Compare Human→AI and AI→Human quality within the same participants."""
    slug = "16_mixed_workflow_direction_quality_practice_rounds"
    plot_df = _phase_data(_prepare_quality_data(df), "practice")
    paired_df = _paired_matrix(plot_df, ["human_ai", "ai_human"])

    _plot_two_condition_paired_comparison(
        paired_df=paired_df,
        left_column="human_ai",
        right_column="ai_human",
        left_label="Human → AI",
        right_label="AI → Human",
        left_color=WORKFLOW_COLORS["human_ai"],
        right_color=WORKFLOW_COLORS["ai_human"],
        slug=slug,
        title="Mixed-Workflow Direction and Quality in Practice Rounds",
        description="Within-participant comparison of the two mixed workflows in "
        "controlled practice rounds. Lines connect the same participant across "
        "Human→AI and AI→Human.",
    )


def plot_mixed_vs_solo_quality_practice_rounds(df: pd.DataFrame) -> None:
    """Compare each participant's mixed- and solo-workflow practice quality."""
    slug = "17_mixed_vs_solo_quality_practice_rounds"
    plot_df = _phase_data(_prepare_quality_data(df), "practice")
    required_workflows = ["human", "ai", "human_ai", "ai_human"]
    matrix = _paired_matrix(plot_df, required_workflows)
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
        title="Mixed versus Solo Workflow Quality in Practice Rounds",
        description="Within-participant comparison of average quality in solo and "
        "mixed workflows. Each participant contributes one average over Human-only "
        "and AI-only outputs and one average over Human→AI and AI→Human outputs.",
    )


# -----------------------------------------------------------------------------
# 18: Main-round quality patterns by actual error exposure
# -----------------------------------------------------------------------------


def _available_exposure_groups(dataframe: pd.DataFrame) -> list[str]:
    """Return present exposure groups in the canonical display order."""
    if "errorExposureGroup" not in dataframe.columns:
        return []

    available = set(dataframe["errorExposureGroup"].dropna().astype(str))
    configured = [group for group in EXPOSURE_LABELS if group in available]
    extras = sorted(available - set(configured))
    return configured + extras


def plot_main_round_quality_by_error_exposure(df: pd.DataFrame) -> None:
    """Plot descriptive quality trends in Main rounds by exposure group."""
    slug = "18_main_round_quality_by_error_exposure"
    plot_df = _phase_data(_prepare_quality_data(df), "main")

    if plot_df.empty:
        return
    if not require_columns(
        plot_df,
        {"roundIndex", "errorExposureGroup"},
        "main-round quality by exposure",
    ):
        return

    plot_df["roundIndex"] = pd.to_numeric(plot_df["roundIndex"], errors="coerce")
    plot_df = plot_df.dropna(subset=["roundIndex", "errorExposureGroup"])
    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)
    plot_df = plot_df[plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)].copy()
    groups = _available_exposure_groups(plot_df)
    if not groups:
        return

    summary = _quality_summary(
        plot_df,
        ["errorExposureGroup", "roundIndex", "workflow"],
    )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.6 * len(groups), 5.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for panel_index, group in enumerate(groups):
        ax = axes[panel_index]
        group_summary = summary[summary["errorExposureGroup"].astype(str) == group]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = group_summary[
                group_summary["workflow"] == workflow
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

        ax.set_title(exposure_display_name(group))
        ax.set_xticks(MAIN_ROUND_INDICES)
        ax.set_xticklabels(
            [f"Main {index}" for index in range(1, len(MAIN_ROUND_INDICES) + 1)]
        )
        ax.set_xlabel("Free-choice main round")
        ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
        if panel_index == 0:
            ax.set_ylabel("Mean composite quality (1–5)")
        apply_standard_axes_style(ax)

    axes[-1].legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "Composite Quality Across Main Rounds by Error-Exposure Group",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(
        fig,
        slug,
        "Composite Quality Across Main Rounds by Error-Exposure Group",
        "Descriptive workflow-level quality trends in free-choice Main rounds, "
        "shown separately for participants exposed and not exposed to the injected "
        "AI error. Exposure groups arise from Main 1 workflow choices and should not "
        "be interpreted as a randomized causal contrast.",
    )


# -----------------------------------------------------------------------------
# 19: Highest- and lowest-rated poems
# -----------------------------------------------------------------------------


def plot_quality_examples(df, max_examples_per_extreme=3):
    """Create optional illustrative examples of highest- and lowest-rated poems.

    These figures are for curious dashboard readers only and are not intended
    as evidence of a general workflow effect.
    """
    required_columns = {QUALITY_PRIMARY_METRIC, "text", "workflow", "roundIndex"}

    if not require_columns(df, required_columns, "quality examples"):
        return

    plot_df = drop_duplicate_participant_rounds(df.copy())
    plot_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        plot_df[QUALITY_PRIMARY_METRIC], errors="coerce"
    )
    plot_df = plot_df.dropna(
        subset=[QUALITY_PRIMARY_METRIC, "text", "workflow", "roundIndex"]
    )

    if plot_df.empty:
        return

    plot_df["text"] = plot_df["text"].astype(str).str.strip()
    plot_df = plot_df[plot_df["text"] != ""]

    if plot_df.empty:
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
        "isFullyRated",
    ]
    export_columns = [column for column in display_columns if column in plot_df.columns]

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

        figure_height = max(5.0, 4.0 * len(example_df))
        fig, ax = plt.subplots(figsize=(11.5, figure_height))
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

        y_position = 0.96
        vertical_gap = 0.90 / len(example_df)

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

            if "isFullyRated" in row and pd.notna(row["isFullyRated"]):
                fully_rated = bool(row["isFullyRated"])
                metadata.append("Fully rated" if fully_rated else "Partially rated")

            header = f"Example {index}  |  " + "  |  ".join(metadata)
            poem_text = wrap_text(row["text"])

            ax.text(
                0.03,
                y_position,
                header,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

            ax.text(
                0.03,
                y_position - 0.06,
                f"Topic: {topic}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                style="italic",
            )

            ax.text(
                0.03,
                y_position - 0.12,
                poem_text,
                transform=ax.transAxes,
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

            y_position -= vertical_gap

        save_figure(fig, slug, title, description)

    highest_examples = select_examples(plot_df, ascending=False)
    lowest_examples = select_examples(plot_df, ascending=True)

    highest_slug = "19_highest_rated_poem_examples"
    lowest_slug = "19b_lowest_rated_poem_examples"

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
        "Illustrative examples selected by the highest composite quality scores. "
        "They are individual cases and should not be interpreted as evidence "
        "that a workflow generally performs better.",
    )

    render_examples(
        lowest_examples,
        lowest_slug,
        "Lowest-Rated Poem Examples",
        "Illustrative examples selected by the lowest composite quality scores. "
        "They are individual cases and should not be interpreted as evidence "
        "that a workflow generally performs worse.",
    )


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------


def plot_quality(df: pd.DataFrame) -> None:
    """Generate the complete output-quality figure set."""
    plot_evaluation_coverage(df)
    plot_composite_quality_by_workflow_all_rounds(df)
    plot_composite_quality_by_workflow_practice_rounds(df)
    plot_quality_dimensions_by_workflow_practice_rounds(df)
    plot_quality_vs_completion_time_by_workflow(df)
    plot_mixed_workflow_direction_quality_practice_rounds(df)
    plot_mixed_vs_solo_quality_practice_rounds(df)
    plot_main_round_quality_by_error_exposure(df)
    plot_quality_examples(df)
