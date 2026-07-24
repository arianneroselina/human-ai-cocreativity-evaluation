"""Pairwise raw-rating matrices for overall quality."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from scripts.config import RATING_SCALE
from scripts.dashboard_figures.helpers import evaluator_display_name
from scripts.utils import save_figure

from scripts.dashboard_figures.evaluator_analysis.common import _pairwise_summary


def plot_pairwise_overall_quality_matrices(
    wide_df: pd.DataFrame,
    evaluators: list[str],
    *,
    pairwise_df: pd.DataFrame | None = None,
) -> None:
    """Show every raw 1-5 rating combination for every evaluator pair."""
    slug = "67_pairwise_overall_quality_rating_matrices"
    if pairwise_df is None:
        pairwise_df = _pairwise_summary(wide_df, evaluators)
    if pairwise_df.empty:
        return

    pairs = list(combinations(evaluators, 2))
    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(5.0 * len(pairs), 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    images = []

    for axis, (evaluator_a, evaluator_b) in zip(axes_flat, pairs):
        matrix = pd.crosstab(
            wide_df[evaluator_b].astype(int),
            wide_df[evaluator_a].astype(int),
        ).reindex(index=RATING_SCALE, columns=RATING_SCALE, fill_value=0)

        matrix_values = matrix.to_numpy()
        image = axis.imshow(
            matrix_values,
            cmap="Blues",
            vmin=0,
            vmax=max(1, int(matrix_values.max())),
            origin="lower",
            aspect="equal",
        )
        images.append(image)

        for row_index, rating_b in enumerate(RATING_SCALE):
            for column_index, rating_a in enumerate(RATING_SCALE):
                count = int(matrix.loc[rating_b, rating_a])
                text_color = "white" if count > matrix_values.max() * 0.55 else "black"
                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

        pair_metrics = pairwise_df[
            pairwise_df["evaluatorA"].eq(evaluator_a)
            & pairwise_df["evaluatorB"].eq(evaluator_b)
        ].iloc[0]

        axis.set_xticks(range(len(RATING_SCALE)))
        axis.set_xticklabels(RATING_SCALE)
        axis.set_yticks(range(len(RATING_SCALE)))
        axis.set_yticklabels(RATING_SCALE)
        axis.set_xlabel(f"{evaluator_display_name(evaluator_a)} rating")
        axis.set_ylabel(f"{evaluator_display_name(evaluator_b)} rating")
        axis.set_title(
            f"{evaluator_display_name(evaluator_a)} vs "
            f"{evaluator_display_name(evaluator_b)}\n"
            f"κw = {pair_metrics['quadraticWeightedKappa']:.3f}"
        )

    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.16,
        top=0.86,
        wspace=0.35,
    )

    colorbar_axis = fig.add_axes([0.91, 0.17, 0.015, 0.68])

    fig.colorbar(
        images[0],
        cax=colorbar_axis,
        label="Number of poems",
    )

    fig.suptitle("Raw Overall-Quality Rating Combinations", fontsize=13, y=0.99)
    fig.text(
        0.01,
        0.01,
        "Diagonal cells show exact agreement. Cells directly next to the diagonal "
        "represent ratings that differed by one point.",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )

    save_figure(
        fig,
        slug,
        "Raw Overall-Quality Rating Combinations",
        "Cross-tabulations of raw 1-5 overall-quality ratings for every evaluator "
        "pair. Diagonal cells show exact agreement; off-diagonal cells show the "
        "direction and magnitude of disagreements.",
    )
