"""Output-quality figures for the randomized practice phase.

Figures
-------
11  Mean overall quality by workflow in practice rounds
12  Quality dimensions by workflow in practice rounds
13  Human→AI versus AI→Human quality in practice rounds
14  Mixed versus solo workflow quality in practice rounds
15  Highest- and lowest-rated practice-round poems
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scripts.config import (
    QUALITY_DIMENSION_LABELS,
    QUALITY_PRIMARY_METRIC,
    WORKFLOW_COLORS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
    QUALITY_Y_MIN,
    QUALITY_Y_MAX,
)
from scripts.dashboard_figures.helpers import (
    phase_data,
    quality_summary,
    workflow_display_name,
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


# -----------------------------------------------------------------------------
# Shared quality-data helpers
# -----------------------------------------------------------------------------


def _prepare_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid observations for the primary quality outcome."""
    if QUALITY_PRIMARY_METRIC not in df.columns:
        return pd.DataFrame()

    prepared = df.copy()
    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    return prepared.dropna(subset=[QUALITY_PRIMARY_METRIC]).copy()


def _workflow_order_present(dataframe: pd.DataFrame) -> list[str]:
    """Return available workflows in the canonical display order."""
    available = set(dataframe["workflow"].dropna().unique())
    return [workflow for workflow in WORKFLOW_ORDER if workflow in available]


def _add_raw_points(
    ax,
    dataframe: pd.DataFrame,
    workflows: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
    seed: int = 42,
) -> None:
    """Add reproducibly jittered poem-level observations."""
    rng = np.random.default_rng(seed)

    for position, workflow in enumerate(workflows, start=1):
        values = (
            dataframe.loc[dataframe["workflow"].eq(workflow), metric]
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
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        if workflow not in indexed_summary.index:
            continue

        row = indexed_summary.loc[workflow]
        mean = float(row["mean"])
        low = row["ciLow"]
        high = row["ciHigh"]

        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
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
    """Annotate workflow plots with observation counts."""
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        if workflow not in indexed_summary.index:
            continue

        ax.text(
            position,
            y_position,
            f"n={int(indexed_summary.loc[workflow, 'count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_workflow_quality_distribution(
    dataframe: pd.DataFrame,
    *,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Create a raw-point, boxplot, and mean/CI workflow comparison."""
    if dataframe.empty:
        return

    workflows = _workflow_order_present(dataframe)
    if not workflows:
        return

    summary = quality_summary(dataframe, ["workflow"])
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    box_data = [
        dataframe.loc[
            dataframe["workflow"].eq(workflow),
            QUALITY_PRIMARY_METRIC,
        ]
        .dropna()
        .to_numpy()
        for workflow in workflows
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
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
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)


def _paired_matrix(
    dataframe: pd.DataFrame,
    workflows: list[str],
) -> pd.DataFrame:
    """Build a complete participant-by-workflow matrix."""
    required = {"participantId", "workflow", QUALITY_PRIMARY_METRIC}
    if not require_columns(dataframe, required, "paired quality comparison"):
        return pd.DataFrame()

    paired_source = dataframe.loc[
        dataframe["workflow"].isin(workflows),
        ["participantId", "workflow", QUALITY_PRIMARY_METRIC],
    ].dropna()

    duplicate_mask = paired_source.duplicated(
        subset=["participantId", "workflow"],
        keep=False,
    )
    if duplicate_mask.any():
        print(
            "Paired quality comparison found duplicate participant-workflow "
            "observations; duplicate values are averaged."
        )

    matrix = (
        paired_source.groupby(
            ["participantId", "workflow"],
            observed=True,
        )[QUALITY_PRIMARY_METRIC]
        .mean()
        .unstack("workflow")
        .reindex(columns=workflows)
    )

    return matrix.dropna(subset=workflows)


def _plot_two_condition_paired_comparison(
    paired_df: pd.DataFrame,
    *,
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
        columns={
            left_column: left_label,
            right_column: right_label,
        }
    )
    save_table(export_df, slug, index=False)

    summary = quality_summary(
        paired_df[[left_column, right_column]].melt(
            var_name="comparison",
            value_name=QUALITY_PRIMARY_METRIC,
        ),
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

        ax.plot(
            x_values,
            y_values,
            color="0.65",
            alpha=0.55,
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            x_values[0],
            y_values[0],
            color=left_color,
            alpha=0.75,
            s=30,
            zorder=2,
        )
        ax.scatter(
            x_values[1],
            y_values[1],
            color=right_color,
            alpha=0.75,
            s=30,
            zorder=2,
        )

    indexed_summary = summary.set_index("comparison")
    for x_position, comparison, color in zip(
        x_positions,
        [left_column, right_column],
        [left_color, right_color],
    ):
        row = indexed_summary.loc[comparison]
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
# 11: Practice-round workflow quality
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# 12: Practice-round dimension-level quality
# -----------------------------------------------------------------------------


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
            "Skipping quality dimensions; no configured dimension columns are available."
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


# -----------------------------------------------------------------------------
# 13-14: Paired practice-round workflow comparisons
# -----------------------------------------------------------------------------


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
        left_label="Human → AI",
        right_label="AI → Human",
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
        left_label="Solo workflows\n(Human-only + AI-only)",
        right_label="Mixed workflows\n(Human→AI + AI→Human)",
        left_color="0.45",
        right_color=WORKFLOW_COLORS["human_ai"],
        slug=slug,
        title="Mixed versus Solo Workflow Quality in Practice Rounds",
        description=(
            "Within-participant comparison of average quality across the two solo "
            "and two mixed workflows in the randomized practice phase."
        ),
    )


# -----------------------------------------------------------------------------
# 15: Highest- and lowest-rated poems
# -----------------------------------------------------------------------------


def plot_quality_examples(prepared, max_examples_per_extreme=3):
    """Create optional illustrative examples of the highest-and lowest-rated poems.

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

    highest_slug = "15_highest_rated_poem_examples"
    lowest_slug = "15b_lowest_rated_poem_examples"

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
    """Generate practice-round output-quality figures."""
    prepared = _prepare_quality_data(df)
    if prepared.empty:
        return

    practice_df = phase_data(prepared, "practice")
    if practice_df.empty:
        return

    plot_overall_quality_by_workflow_practice_rounds(practice_df)
    plot_rating_dimensions_by_workflow_practice_rounds(practice_df)
    plot_mixed_workflow_direction_quality_practice_rounds(practice_df)
    plot_mixed_vs_solo_quality_practice_rounds(practice_df)
    plot_quality_examples(prepared)
