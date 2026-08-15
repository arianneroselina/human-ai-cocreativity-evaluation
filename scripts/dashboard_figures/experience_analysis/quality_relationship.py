"""Relationship between participant satisfaction and external quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    PHASES,
    QUALITY_PRIMARY_METRIC,
    QUALITY_SCALE_MAX,
    QUALITY_SCALE_MIN,
    SATISFACTION_COLUMN,
)
from scripts.dashboard_figures.style import apply_standard_axes_style, SUBTITLE_FONT_SIZE
from scripts.utils import require_columns, save_figure, save_table

from scripts.dashboard_figures.experience_analysis.common import _spearman_summary


def plot_satisfaction_vs_external_quality(prepared) -> None:
    """Show how external quality varies across participant satisfaction levels.

    Satisfaction is treated as an ordered 1-5 response scale. Within each
    study phase, the figure shows the distribution of external quality for
    each satisfaction level, together with raw participant-round observations
    and the median quality trend.
    """
    slug = "34_satisfaction_vs_external_quality"

    required_columns = {
        "phase",
        "workflow",
        "roundIndex",
        SATISFACTION_COLUMN,
        QUALITY_PRIMARY_METRIC,
    }
    if not require_columns(
        prepared,
        required_columns,
        "satisfaction versus external quality",
    ):
        return

    plot_df = prepared.copy()

    plot_df[SATISFACTION_COLUMN] = pd.to_numeric(
        plot_df[SATISFACTION_COLUMN],
        errors="coerce",
    )
    plot_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        plot_df[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "phase",
            SATISFACTION_COLUMN,
            QUALITY_PRIMARY_METRIC,
        ]
    )

    plot_df = plot_df[
        plot_df[SATISFACTION_COLUMN].between(1, 5)
        & plot_df[QUALITY_PRIMARY_METRIC].between(
            QUALITY_SCALE_MIN,
            QUALITY_SCALE_MAX,
        )
    ].copy()

    if plot_df.empty:
        return

    available_phases = [
        phase for phase in PHASES if phase in set(plot_df["phase"].dropna())
    ]

    if not available_phases:
        return

    statistics = pd.DataFrame(
        [_spearman_summary(plot_df, phase) for phase in available_phases]
    )
    save_table(statistics, slug, index=False)

    satisfaction_levels = np.arange(1, 6)

    fig, axes = plt.subplots(
        1,
        len(available_phases),
        figsize=(7.0 * len(available_phases), 5.6),
        sharey=True,
        squeeze=False,
    )

    for phase_index, (axis, phase) in enumerate(zip(axes.flatten(), available_phases)):
        phase_df = plot_df[plot_df["phase"].eq(phase)].copy()

        box_data = []
        box_positions = []
        counts = []
        median_x = []
        median_y = []

        rng = np.random.default_rng(700 + phase_index)

        for satisfaction_level in satisfaction_levels:
            quality_values = phase_df.loc[
                phase_df[SATISFACTION_COLUMN].eq(satisfaction_level),
                QUALITY_PRIMARY_METRIC,
            ].dropna()

            counts.append(len(quality_values))

            if quality_values.empty:
                continue

            box_data.append(quality_values.to_numpy(dtype=float))
            box_positions.append(satisfaction_level)

            # Raw observations, horizontally jittered only to prevent overlap.
            x_jitter = satisfaction_level + rng.uniform(
                -0.16,
                0.16,
                size=len(quality_values),
            )

            axis.scatter(
                x_jitter,
                quality_values,
                s=26,
                color="#777777",
                alpha=0.38,
                edgecolor="none",
                zorder=2,
            )

            median_x.append(satisfaction_level)
            median_y.append(float(quality_values.median()))

        if box_data:
            boxplot = axis.boxplot(
                box_data,
                positions=box_positions,
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                medianprops={
                    "color": "#222222",
                    "linewidth": 1.8,
                },
                boxprops={
                    "edgecolor": "#555555",
                    "linewidth": 1.1,
                },
                whiskerprops={
                    "color": "#555555",
                    "linewidth": 1.0,
                },
                capprops={
                    "color": "#555555",
                    "linewidth": 1.0,
                },
            )

            for box in boxplot["boxes"]:
                box.set_facecolor("#e6e6e6")
                box.set_alpha(0.95)

        # Connect medians to make the overall descriptive pattern easy to see.
        if len(median_x) >= 2:
            axis.plot(
                median_x,
                median_y,
                color="#222222",
                linewidth=1.6,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="#222222",
                zorder=4,
            )

        phase_stat = statistics.loc[statistics["phase"].eq(phase)].iloc[0]

        rho = phase_stat.get("spearmanRho")
        observations = int(phase_stat.get("observations", 0))

        statistic_text = (
            f"Spearman ρ = {rho:.2f}\nn = {observations}"
            if pd.notna(rho)
            else f"n = {observations}"
        )

        axis.text(
            0.04,
            0.96,
            statistic_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=SUBTITLE_FONT_SIZE,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.95,
            },
        )

        axis.set_title(
            f"{phase.capitalize()} rounds",
            fontsize=11,
            fontweight="bold",
        )

        axis.set_xlim(0.5, 5.5)
        axis.set_ylim(QUALITY_SCALE_MIN - 0.3, QUALITY_SCALE_MAX + 0.3)

        axis.set_xticks(satisfaction_levels)
        axis.set_xticklabels(
            [
                f"{level}\n(n={count})"
                for level, count in zip(satisfaction_levels, counts)
            ]
        )

        axis.set_yticks(
            np.arange(
                int(QUALITY_SCALE_MIN),
                int(QUALITY_SCALE_MAX) + 1,
            )
        )

        axis.set_xlabel("Participant satisfaction rating\n(n shown below each rating)")
        axis.set_ylabel("External quality composite (1-5)")

        apply_standard_axes_style(axis, grid_axis="y")

    raw_points_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#777777",
        markersize=7,
        alpha=0.55,
        label="Individual participant-round",
    )

    median_handle = plt.Line2D(
        [0],
        [0],
        color="#222222",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#222222",
        linewidth=1.6,
        markersize=5,
        label="Median external quality",
    )

    fig.legend(
        handles=[raw_points_handle, median_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "External Text Quality by Participant Satisfaction",
        fontsize=13,
        y=0.995,
    )

    fig.text(
        0.02,
        0.085,
        "Boxes show the interquartile range; horizontal lines inside boxes show "
        "the median. The connected median line is descriptive only. Spearman "
        "correlations pool workflows within each study phase.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4a4a4a",
    )

    fig.tight_layout(rect=(0, 0.13, 1, 0.97))

    save_figure(
        fig,
        slug,
        "External Text Quality by Participant Satisfaction",
        "Distribution of external quality-composite scores at each participant "
        "satisfaction level, shown separately for Practice and Main rounds. "
        "Boxes show interquartile ranges, individual points show participant-rounds, "
        "and connected markers show median external quality. Spearman correlations "
        "are descriptive associations pooling workflows within each study phase.",
    )
