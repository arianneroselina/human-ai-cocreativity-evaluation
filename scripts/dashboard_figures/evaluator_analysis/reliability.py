"""ICC and pairwise evaluator reliability figures."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import (
    save_figure,
    save_table,
)

from scripts.dashboard_figures.evaluator_analysis.common import (
    _icc_summary,
    _pairwise_summary,
)


def plot_overall_quality_icc_reliability(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Plot absolute-agreement ICC for one evaluator and the evaluator mean."""
    slug = "63_overall_quality_icc_reliability"

    icc_df = _icc_summary(wide_df)
    if icc_df.empty:
        return

    save_table(
        icc_df,
        f"{slug}_summary",
        index=False,
    )

    icc_plot = (
        icc_df.set_index("statistic")
        .reindex(["ICC(A,1)", "ICC(A,k)"])
        .dropna(subset=["icc"])
        .reset_index()
    )
    if icc_plot.empty:
        return

    y_positions = np.arange(len(icc_plot))
    values = icc_plot["icc"].to_numpy(dtype=float)
    lower_ci = icc_plot["lowerCI"].to_numpy(dtype=float)
    upper_ci = icc_plot["upperCI"].to_numpy(dtype=float)

    lower_error = np.maximum(values - lower_ci, 0.0)
    upper_error = np.maximum(upper_ci - values, 0.0)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    ax.errorbar(
        values,
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color="#333333",
        markersize=9,
        capsize=5,
        linewidth=1.7,
        zorder=3,
    )

    for y_position, (_, row) in zip(y_positions, icc_plot.iterrows()):
        ax.annotate(
            (f"{row['icc']:.3f}\n95% CI [{row['lowerCI']:.2f}, {row['upperCI']:.2f}]"),
            (row["icc"], y_position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlim(-0.05, 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            "One evaluator\nICC(A,1)",
            f"Mean of {len(evaluators)} evaluators\nICC(A,k)",
        ]
    )
    ax.set_xlabel("Absolute-agreement intraclass correlation")
    ax.set_title("Overall-Quality ICC Reliability")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            f"ICC(A,1) describes one evaluator's score; ICC(A,k) describes the "
            f"mean score across {len(evaluators)} evaluators for {len(wide_df)} poems."
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
        "Overall-Quality ICC Reliability",
        (
            "Absolute-agreement ICC values and 95% confidence intervals for one "
            "evaluator and the mean across the full evaluator panel."
        ),
    )


def plot_pairwise_overall_quality_agreement(
    wide_df: pd.DataFrame,
    evaluators: list[str],
    *,
    pairwise_df: pd.DataFrame | None = None,
) -> None:
    """Plot pairwise weighted kappa, exact agreement, and one-point agreement."""
    slug = "64_pairwise_overall_quality_agreement"

    if pairwise_df is None:
        pairwise_df = _pairwise_summary(wide_df, evaluators)
    if pairwise_df.empty:
        return

    save_table(
        pairwise_df,
        f"{slug}_summary",
        index=False,
    )

    pairwise_plot = pairwise_df.copy()
    pairwise_plot["containsAI"] = pairwise_plot["evaluatorALabel"].str.contains(
        "AI",
        case=False,
        na=False,
    ) | pairwise_plot["evaluatorBLabel"].str.contains(
        "AI",
        case=False,
        na=False,
    )

    # Human-human comparison first, followed by the two AI-involving pairs.
    pairwise_plot = pairwise_plot.sort_values(
        ["containsAI", "pairLabel"],
        ascending=[True, True],
    ).reset_index(drop=True)

    y_positions = np.arange(len(pairwise_plot))
    bar_colours = [
        "#9a5f53" if contains_ai else "#888888"
        for contains_ai in pairwise_plot["containsAI"]
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.6))

    ax.barh(
        y_positions,
        pairwise_plot["quadraticWeightedKappa"],
        color=bar_colours,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    for y_position, (_, row) in zip(
        y_positions,
        pairwise_plot.iterrows(),
    ):
        ax.text(
            row["quadraticWeightedKappa"] + 0.012,
            y_position,
            (
                f"κw={row['quadraticWeightedKappa']:.3f}  |  "
                f"exact {row['exactAgreementPercentage']:.1f}%  |  "
                f"±1 {row['withinOnePointPercentage']:.1f}%"
            ),
            va="center",
            fontsize=8.4,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlim(-0.03, 1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pairwise_plot["pairLabel"])
    ax.invert_yaxis()
    ax.set_xlabel("Quadratic-weighted Cohen's kappa")
    ax.set_title("Pairwise Overall-Quality Agreement")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            f"Pairwise agreement across {len(wide_df)} shared poems. "
            "Brown bars involve the AI evaluator; the grey bar compares the two humans."
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
        "Pairwise Overall-Quality Agreement",
        (
            "Pairwise quadratic-weighted Cohen's kappa, exact agreement, and "
            "within-one-point agreement for every evaluator pair."
        ),
    )
