"""Illustrative highest- and lowest-rated poem examples."""

from __future__ import annotations

import textwrap

import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    QUALITY_PRIMARY_METRIC,
    WORKFLOW_LABELS,
)
from scripts.utils import require_columns, save_figure, save_table


def plot_quality_examples(prepared, max_examples_per_extreme=3):
    """Create one compact figure with highest- and lowest-rated poems."""

    required_columns = {
        QUALITY_PRIMARY_METRIC,
        "text",
        "workflow",
        "roundIndex",
    }

    if not require_columns(prepared, required_columns, "quality examples"):
        return

    prepared = prepared.copy()

    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    prepared = prepared.dropna(
        subset=[
            QUALITY_PRIMARY_METRIC,
            "text",
            "workflow",
            "roundIndex",
        ]
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

    def wrap_text(value, width):
        """Wrap text while preserving existing line breaks."""

        wrapped_lines = []

        for original_line in str(value).splitlines():
            line = original_line.strip()

            if not line:
                wrapped_lines.append("")
                continue

            wrapped = textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )

            wrapped_lines.extend(wrapped if wrapped else [""])

        return "\n".join(wrapped_lines)

    def count_lines(value):
        return str(value).count("\n") + 1

    def select_examples(dataframe, ascending=False):
        sorted_df = dataframe.sort_values(
            by=[
                QUALITY_PRIMARY_METRIC,
                "roundIndex",
            ],
            ascending=[
                ascending,
                True,
            ],
            kind="stable",
        )

        return sorted_df.head(max_examples_per_extreme).copy()

    highest_examples = select_examples(
        prepared,
        ascending=False,
    )

    lowest_examples = select_examples(
        prepared,
        ascending=True,
    )

    save_table(
        highest_examples[export_columns],
        "15_highest_rated_poem_examples",
        index=False,
    )

    save_table(
        lowest_examples[export_columns],
        "15b_lowest_rated_poem_examples",
        index=False,
    )

    # ------------------------------------------------------------
    # Prepare examples before creating the figure
    # ------------------------------------------------------------

    def prepare_examples(example_df):
        example_df = example_df.copy()

        example_df["workflowLabel"] = (
            example_df["workflow"].map(WORKFLOW_LABELS).fillna(example_df["workflow"])
        )

        examples = []

        for _, row in example_df.iterrows():
            quality = float(row[QUALITY_PRIMARY_METRIC])

            metadata = [
                f"Round {int(row['roundIndex'])}",
                str(row["workflowLabel"]),
                f"Quality {quality:.2f}/5.00",
            ]

            if "constraintScore" in row and pd.notna(row["constraintScore"]):
                metadata.append(
                    f"Constraint {float(row['constraintScore']):.2f}/100.00"
                )

            header = wrap_text(
                " | ".join(metadata),
                width=58,
            )

            topic = row.get("topic")

            if pd.isna(topic):
                topic = "Topic unavailable"

            topic_text = wrap_text(
                f"Topic: {topic}",
                width=55,
            )

            poem_text = wrap_text(
                row["text"],
                width=58,
            )

            examples.append(
                {
                    "header": header,
                    "topic": topic_text,
                    "poem": poem_text,
                }
            )

        return examples

    highest = prepare_examples(highest_examples)
    lowest = prepare_examples(lowest_examples)

    # ------------------------------------------------------------
    # Compact physical spacing in inches
    # ------------------------------------------------------------

    FIGURE_WIDTH = 7.2

    # Text line heights
    HEADER_LINE_HEIGHT = 0.105
    TOPIC_LINE_HEIGHT = 0.105
    POEM_LINE_HEIGHT = 0.100

    # Very small gaps
    HEADER_TOPIC_GAP = 0.025
    TOPIC_POEM_GAP = 0.035
    EXAMPLE_GAP = 0.085

    # Figure-level spacing
    TOP_MARGIN = 0.12
    COLUMN_TITLE_HEIGHT = 0.17
    COLUMN_TITLE_GAP = 0.07
    BOTTOM_MARGIN = 0.10

    def example_height(example):
        return (
            count_lines(example["header"]) * HEADER_LINE_HEIGHT
            + HEADER_TOPIC_GAP
            + count_lines(example["topic"]) * TOPIC_LINE_HEIGHT
            + TOPIC_POEM_GAP
            + count_lines(example["poem"]) * POEM_LINE_HEIGHT
            + EXAMPLE_GAP
        )

    def column_height(examples):
        return sum(example_height(example) for example in examples)

    content_height = max(
        column_height(highest),
        column_height(lowest),
    )

    # Height is determined by the actual content.
    FIGURE_HEIGHT = (
        TOP_MARGIN
        + COLUMN_TITLE_HEIGHT
        + COLUMN_TITLE_GAP
        + content_height
        + BOTTOM_MARGIN
    )

    fig = plt.figure(
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
    )

    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------
    # Helpers for inch-based positioning
    # ------------------------------------------------------------

    def y_fraction(y_inches):
        return 1 - (y_inches / FIGURE_HEIGHT)

    # ------------------------------------------------------------
    # Column headings
    # ------------------------------------------------------------

    current_top = TOP_MARGIN

    fig.text(
        0.015,
        y_fraction(current_top),
        "Highest-rated",
        ha="left",
        va="top",
        fontsize=8,
    )

    fig.text(
        0.515,
        y_fraction(current_top),
        "Lowest-rated",
        ha="left",
        va="top",
        fontsize=8,
    )

    content_start = TOP_MARGIN + COLUMN_TITLE_HEIGHT + COLUMN_TITLE_GAP

    # ------------------------------------------------------------
    # Draw one column
    # ------------------------------------------------------------

    def draw_column(examples, x_position):
        y = content_start

        for example in examples:
            # Metadata
            fig.text(
                x_position,
                y_fraction(y),
                example["header"],
                ha="left",
                va="top",
                fontsize=6.4,
            )

            y += count_lines(example["header"]) * HEADER_LINE_HEIGHT

            y += HEADER_TOPIC_GAP

            # Topic
            fig.text(
                x_position,
                y_fraction(y),
                example["topic"],
                ha="left",
                va="top",
                fontsize=6.5,
                style="italic",
            )

            y += count_lines(example["topic"]) * TOPIC_LINE_HEIGHT

            y += TOPIC_POEM_GAP

            # Poem
            fig.text(
                x_position,
                y_fraction(y),
                example["poem"],
                ha="left",
                va="top",
                fontsize=6.5,
                family="monospace",
                linespacing=1.0,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#d9d9d9",
                    "linewidth": 0.5,
                },
            )

            y += count_lines(example["poem"]) * POEM_LINE_HEIGHT

            y += EXAMPLE_GAP

    draw_column(
        highest,
        x_position=0.015,
    )

    draw_column(
        lowest,
        x_position=0.515,
    )

    save_figure(
        fig,
        "15_quality_extreme_poem_examples",
        "Highest- and Lowest-Rated Poem Examples",
        (
            "Illustrative examples selected from the highest and lowest "
            "Mean Overall Quality scores. Individual examples should not "
            "be interpreted as general workflow effects."
        ),
    )
