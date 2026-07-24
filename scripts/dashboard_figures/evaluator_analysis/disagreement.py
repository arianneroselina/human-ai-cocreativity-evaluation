"""Evaluator disagreement magnitude and rating tendency."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.dashboard_figures.helpers import evaluator_color
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import (
    save_figure,
    save_table,
)

from scripts.dashboard_figures.evaluator_analysis.common import (
    _disagreement_outputs,
    _tendency_summary,
)


def plot_evaluator_disagreement_magnitude(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Show the poem-level spread across evaluator ratings."""
    slug = "65_evaluator_disagreement_magnitude"
    range_df, disagreement_rows = _disagreement_outputs(wide_df, evaluators)

    if range_df.empty:
        return

    save_table(
        range_df,
        f"{slug}_distribution",
        index=False,
    )
    save_table(
        disagreement_rows,
        f"{slug}_poem_level",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    bars = ax.bar(
        range_df["ratingRange"].astype(str),
        range_df["percentage"],
        color="#8BAE66",
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    for bar, (_, row) in zip(bars, range_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{int(row['poemCount'])}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_ylim(0, max(5, float(range_df["percentage"].max()) + 12))
    ax.set_xlabel("Rating range across all evaluators")
    ax.set_ylabel("Poems (%)")
    ax.set_title("Magnitude of Evaluator Disagreement")
    apply_standard_axes_style(ax, grid_axis="y")

    fig.text(
        0.01,
        0.01,
        (
            "Rating range = highest minus lowest raw 1-5 rating for the same poem. "
            "A range of 0 indicates exact agreement."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Magnitude of Evaluator Disagreement",
        (
            "Distribution of the difference between the highest and lowest "
            "overall-quality rating assigned to each poem."
        ),
    )


def plot_evaluator_rating_tendency(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Show systematic evaluator strictness or generosity."""
    slug = "66_evaluator_rating_tendency"
    tendency_df = _tendency_summary(wide_df, evaluators)

    if tendency_df.empty:
        return

    save_table(
        tendency_df,
        f"{slug}_summary",
        index=False,
    )

    tendency_plot = tendency_df.reset_index(drop=True)
    y_positions = np.arange(len(tendency_plot))
    deviations = tendency_plot["meanDeviationFromPeers"].to_numpy(dtype=float)
    lower_error = deviations - tendency_plot["lowerCI"].to_numpy(dtype=float)
    upper_error = tendency_plot["upperCI"].to_numpy(dtype=float) - deviations

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.errorbar(
        deviations,
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="none",
        capsize=4,
        linewidth=1.4,
        color="#333333",
        zorder=2,
    )

    for y_position, (_, row) in zip(y_positions, tendency_plot.iterrows()):
        ax.scatter(
            row["meanDeviationFromPeers"],
            y_position,
            s=70,
            color=evaluator_color(row["evaluatorId"]),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            (
                f"Mean rating {row['meanRating']:.2f}\n"
                f"Peer deviation {row['meanDeviationFromPeers']:+.2f}"
            ),
            (row["meanDeviationFromPeers"], y_position),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )

    max_abs = max(
        0.3,
        abs(float(tendency_plot["lowerCI"].min())),
        abs(float(tendency_plot["upperCI"].max())),
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tendency_plot["evaluatorLabel"])
    ax.set_xlabel("Mean deviation from the other evaluators")
    ax.set_ylabel("Evaluator")
    ax.set_title("Relative Evaluator Rating Tendency")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            "Positive values indicate more generous ratings than the other "
            "evaluators; negative values indicate stricter ratings."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Relative Evaluator Rating Tendency",
        (
            "Mean deviation of each evaluator's rating from the average rating "
            "of the other evaluators, with 95% confidence intervals."
        ),
    )
