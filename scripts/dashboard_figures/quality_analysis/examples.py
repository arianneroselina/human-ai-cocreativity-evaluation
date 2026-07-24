"""Illustrative highest- and lowest-rated poem examples."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    QUALITY_PRIMARY_METRIC,
    WORKFLOW_LABELS,
)
from scripts.utils import require_columns, save_figure, save_table


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
