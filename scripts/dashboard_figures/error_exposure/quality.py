"""Main-round external quality by injected-error exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    MAIN_ROUND_INDICES,
    QUALITY_Y_MAX,
    QUALITY_Y_MIN,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.error_exposure.rate_plot import (
    add_injected_error_round_marker,
    _is_exposed,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    round_tick_labels,
    ordered_exposure_groups,
    quality_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.style import WORKFLOW_COLORS, apply_standard_axes_style
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


def plot_main_round_quality_by_error_exposure(
    main_df: pd.DataFrame,
) -> None:
    """Show Main round quality by exposure group and selected workflow.

    Each point represents a separate exposure-group, Main-round, and workflow
    mean. Points are not connected because participants could change workflows
    between Main rounds.
    """
    slug = "106_main_round_quality_by_error_exposure"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
    }
    if not require_columns(
        main_df,
        required,
        "Main round quality by exposure",
    ):
        return

    plot_df = main_df.copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
        ]
    )

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
    ].copy()

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)

    rounds = [
        round_index
        for round_index in MAIN_ROUND_INDICES
        if round_index in plot_df["roundIndex"].unique()
    ]

    if not groups or not rounds:
        return

    summary = quality_summary(
        plot_df,
        [
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
    )

    if summary.empty:
        return

    # Add counts separately if quality_summary does not already return them.
    if "count" not in summary.columns:
        counts = (
            plot_df.groupby(
                [
                    "errorExposed",
                    "roundIndex",
                    "workflow",
                ],
                dropna=False,
            )
            .size()
            .rename("count")
            .reset_index()
        )

        summary = summary.merge(
            counts,
            on=[
                "errorExposed",
                "roundIndex",
                "workflow",
            ],
            how="left",
        )

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)

    save_table(summary, slug, index=False)

    # Separate workflow estimates slightly around each Main-round position.
    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(
                -0.18,
                0.18,
                len(WORKFLOW_ORDER),
            ),
        )
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.6 * len(groups), 5.3),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[summary["errorExposed"].eq(group)]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[group_summary["workflow"].eq(workflow)]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            means = pd.to_numeric(
                workflow_summary["mean"],
                errors="coerce",
            ).to_numpy(dtype=float)

            lows = pd.to_numeric(
                workflow_summary["ciLow"],
                errors="coerce",
            ).to_numpy(dtype=float)

            highs = pd.to_numeric(
                workflow_summary["ciHigh"],
                errors="coerce",
            ).to_numpy(dtype=float)

            counts = (
                pd.to_numeric(
                    workflow_summary["count"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            valid = np.isfinite(means)

            if not valid.any():
                continue

            x_values = np.asarray(rounds, dtype=float) + workflow_offsets[workflow]

            # Mean dots without connecting lines.
            axis.scatter(
                x_values[valid],
                means[valid],
                s=52,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.8,
                label=workflow_display_name(workflow),
                zorder=4,
            )

            valid_ci = valid & np.isfinite(lows) & np.isfinite(highs)

            if valid_ci.any():
                clipped_lows = np.clip(
                    lows[valid_ci],
                    QUALITY_Y_MIN,
                    QUALITY_Y_MAX,
                )
                clipped_highs = np.clip(
                    highs[valid_ci],
                    QUALITY_Y_MIN,
                    QUALITY_Y_MAX,
                )

                axis.errorbar(
                    x_values[valid_ci],
                    means[valid_ci],
                    yerr=np.vstack(
                        [
                            means[valid_ci] - clipped_lows,
                            clipped_highs - means[valid_ci],
                        ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.1,
                    capsize=3,
                    capthick=1.0,
                    alpha=0.9,
                    zorder=3,
                )

            # Add the sample size above each point.
            for x_value, mean, count in zip(
                x_values[valid],
                means[valid],
                counts[valid],
            ):
                axis.annotate(
                    f"n={count}",
                    (x_value, mean),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                    zorder=5,
                )

        if _is_exposed(group):
            add_injected_error_round_marker(axis, rounds)

        axis.set_title(exposure_display_name(group))
        axis.set_xticks(rounds)
        axis.set_xticklabels(round_tick_labels(rounds))
        axis.set_xlim(
            min(rounds) - 0.45,
            max(rounds) + 0.45,
        )

        axis.set_ylim(
            QUALITY_Y_MIN - 0.1,
            QUALITY_Y_MAX + 0.35,
        )

        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel("Mean overall quality (1–5)")

    # Build one shared legend from all panels because some workflows may
    # not appear in one of the exposure groups.
    legend_items = {}

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="Workflow selected",
            bbox_to_anchor=(0.87, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "Mean Overall Quality by Main Round, Workflow, and Error Exposure",
        fontsize=13,
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0.065, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Mean Overall Quality by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean overall-quality estimates for each Main round, "
            "selected workflow, and actual error-exposure group. Points are "
            "not connected because workflow membership could change between "
            "rounds. Whiskers show approximate 95% confidence intervals and "
            "labels show observation counts."
        ),
    )
