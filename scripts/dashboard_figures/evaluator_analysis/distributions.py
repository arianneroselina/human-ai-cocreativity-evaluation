"""Evaluator rating distributions and ordinal agreement."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import RATING_SCALE
from scripts.dashboard_figures.helpers import (
    evaluator_color,
    evaluator_display_name,
)
from scripts.dashboard_figures.style import apply_standard_axes_style, VALUE_LABEL_FONT_SIZE
from scripts.utils import (
    save_figure,
    save_table,
)

from scripts.dashboard_figures.evaluator_analysis.common import (
    _ordinal_krippendorff_alpha_summary,
    _rating_distribution,
)


def plot_overall_quality_rating_distribution(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Show the raw 1-5 overall-quality distribution of each evaluator."""
    slug = "61_overall_quality_rating_distribution"
    distribution_df = _rating_distribution(wide_df, evaluators)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(RATING_SCALE))
    width = 0.76 / len(evaluators)
    offsets = np.linspace(
        -(len(evaluators) - 1) * width / 2,
        (len(evaluators) - 1) * width / 2,
        len(evaluators),
    )

    for offset, evaluator in zip(offsets, evaluators):
        subset = (
            distribution_df[distribution_df["evaluatorId"].eq(evaluator)]
            .set_index("rating")
            .reindex(RATING_SCALE)
        )
        bars = ax.bar(
            positions + offset,
            subset["percentage"].to_numpy(dtype=float),
            width=width,
            color=evaluator_color(evaluator),
            edgecolor="white",
            linewidth=0.8,
            label=evaluator_display_name(evaluator),
            zorder=2,
        )

        for bar, count in zip(bars, subset["poemCount"].to_numpy(dtype=int)):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_LABEL_FONT_SIZE,
                )

    ax.set_xticks(positions)
    ax.set_xticklabels(RATING_SCALE)
    ax.set_ylim(0, max(5, float(distribution_df["percentage"].max()) + 10))
    ax.set_xlabel("Raw overall-quality rating")
    ax.set_ylabel("Rated poems (%)")
    ax.set_title("Overall-Quality Rating Distribution by Evaluator")
    ax.legend(title="Official evaluator")
    apply_standard_axes_style(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        slug,
        "Overall-Quality Rating Distribution by Evaluator",
        "Raw 1-5 overall-quality ratings assigned to the same complete set of "
        "poems by each official evaluator. Labels show poem counts.",
    )


def plot_overall_quality_ordinal_agreement(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Plot ordinal Krippendorff's alpha across the full evaluator panel."""
    slug = "62_overall_quality_ordinal_agreement"

    alpha_df = _ordinal_krippendorff_alpha_summary(
        wide_df,
        evaluators,
    )
    if alpha_df.empty:
        return

    save_table(
        alpha_df,
        f"{slug}_summary",
        index=False,
    )

    alpha_row = alpha_df.iloc[0]
    alpha = float(alpha_row["alpha"])
    lower_ci = float(alpha_row["lowerCI"])
    upper_ci = float(alpha_row["upperCI"])

    fig, ax = plt.subplots(figsize=(7.4, 3.8))

    if np.isfinite(lower_ci) and np.isfinite(upper_ci):
        lower_error = max(alpha - lower_ci, 0.0)
        upper_error = max(upper_ci - alpha, 0.0)

        ax.errorbar(
            [alpha],
            [0],
            xerr=np.array([[lower_error], [upper_error]]),
            fmt="o",
            color="#333333",
            markersize=10,
            capsize=5,
            linewidth=1.8,
            zorder=3,
        )
        ci_label = f"95% CI [{lower_ci:.2f}, {upper_ci:.2f}]"
    else:
        ax.scatter(
            [alpha],
            [0],
            s=100,
            color="#333333",
            zorder=3,
        )
        ci_label = "95% CI unavailable"

    ax.annotate(
        f"αordinal = {alpha:.3f}\n{ci_label}",
        (alpha, 0),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        fontsize=VALUE_LABEL_FONT_SIZE,
    )

    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([0])
    ax.set_yticklabels([f"All {len(evaluators)} evaluators"])
    ax.set_xlabel("Ordinal Krippendorff's alpha")
    ax.set_title("Overall-Quality Ordinal Inter-Rater Agreement")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Overall-Quality Ordinal Inter-Rater Agreement",
        (
            "Ordinal Krippendorff's alpha and bootstrap 95% confidence interval "
            "across all official evaluators."
        ),
    )
