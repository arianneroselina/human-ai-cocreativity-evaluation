"""Main-round NASA-TLX workload by exposure and workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.error_exposure.rate_plot import (
    add_injected_error_round_marker,
    _is_exposed,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    round_display_name,
    ordered_exposure_groups,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    apply_standard_axes_style,
    SUBTITLE_FONT_SIZE, INSIDE_LABEL_FONT_SIZE,
)
from scripts.dashboard_figures.summaries import grouped_metric_summary
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


def plot_main_round_tlx_by_exposure_and_workflow(
    main_df: pd.DataFrame,
) -> None:
    """Show Raw NASA-TLX by Main round, error exposure, and selected workflow.

    Each point represents a separate exposure, Main round, and workflow cell.
    Points are not connected because participants could switch workflows between
    Main rounds.
    """
    slug = "104_main_round_tlx_by_exposure_round_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "rawNasaTlxScore",
    }
    if not require_columns(
        main_df,
        required,
        "Raw NASA-TLX in Main Rounds by error exposure",
    ):
        return

    plot_df = main_df[
        [
            "roundIndex",
            "workflow",
            "errorExposed",
            "rawNasaTlxScore",
        ]
    ].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )
    plot_df["rawNasaTlxScore"] = pd.to_numeric(
        plot_df["rawNasaTlxScore"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            "rawNasaTlxScore",
        ]
    )
    plot_df = plot_df.loc[
        plot_df["workflow"].isin(WORKFLOW_ORDER)
        & plot_df["rawNasaTlxScore"].between(0, 20)
    ].copy()

    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)

    if plot_df.empty:
        return

    groups = ordered_exposure_groups(plot_df)
    rounds = sorted(plot_df["roundIndex"].unique().tolist())

    if not groups or not rounds:
        return

    summary = grouped_metric_summary(
        plot_df,
        group_columns=[
            "errorExposed",
            "roundIndex",
            "workflow",
        ],
        metric_columns=["rawNasaTlxScore"],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)

    save_table(summary, slug, index=False)

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
        figsize=(6.7 * len(groups), 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = summary.loc[
            summary["errorExposed"].eq(group) & summary["metric"].eq("rawNasaTlxScore")
        ]

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[group_summary["workflow"].eq(workflow)]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            valid = workflow_summary["mean"].notna().to_numpy()
            if not valid.any():
                continue

            x_values = np.asarray(rounds, dtype=float) + workflow_offsets[workflow]
            means = workflow_summary["mean"].to_numpy(dtype=float)

            counts = (
                pd.to_numeric(
                    workflow_summary["count"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            # Separate points without connecting lines.
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

            # Confidence intervals are shown only when at least three
            # observations support the cell estimate.
            ci_valid = (
                workflow_summary["mean"].notna()
                & workflow_summary["lowerCI"].notna()
                & workflow_summary["upperCI"].notna()
                & workflow_summary["count"].ge(3)
            ).to_numpy()

            if ci_valid.any():
                ci_rows = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna()
                    & workflow_summary["count"].ge(3)
                ]

                lower_ci = np.clip(
                    ci_rows["lowerCI"].to_numpy(dtype=float),
                    0,
                    20,
                )
                upper_ci = np.clip(
                    ci_rows["upperCI"].to_numpy(dtype=float),
                    0,
                    20,
                )

                axis.errorbar(
                    x_values[ci_valid],
                    means[ci_valid],
                    yerr=np.vstack(
                        [
                            means[ci_valid] - lower_ci,
                            upper_ci - means[ci_valid],
                        ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=2,
                    capsize=3,
                    alpha=0.9,
                    zorder=3,
                )

            # Show the observation count for every cell.
            for x_value, mean, count in zip(
                    x_values[valid],
                    means[valid],
                    counts[valid],
            ):
                # Put labels below points close to the upper boundary.
                if mean >= 17.5:
                    text_offset = (0, -10)
                    vertical_alignment = "top"
                else:
                    text_offset = (0, 9)
                    vertical_alignment = "bottom"

                axis.annotate(
                    f"n={count}",
                    (x_value, mean),
                    xytext=text_offset,
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=INSIDE_LABEL_FONT_SIZE,
                    color="0.25",
                    zorder=5,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.80,
                    },
                )

        round_labels = [
            round_display_name(round_index) for index, round_index in enumerate(rounds)
        ]

        if _is_exposed(group):
            add_injected_error_round_marker(axis, rounds)

        axis.set_xticks(rounds)
        axis.set_xticklabels(round_labels)
        axis.set_xlim(
            min(rounds) - 0.45,
            max(rounds) + 0.45,
        )
        axis.set_ylim(-0.5, 20.8)
        axis.set_yticks([0, 5, 10, 15, 20])

        axis.set_title(exposure_display_name(group))
        axis.set_xlabel("Main round")

        apply_standard_axes_style(
            axis,
            grid_axis="y",
        )

    axes[0].set_ylabel("Raw NASA-TLX workload score (0–20)")

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
        "Raw NASA-TLX Workload in Main Rounds by Error Exposure",
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0.065, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Raw NASA-TLX Workload in Main Rounds by Error Exposure",
        (
            "Separate mean Raw NASA-TLX estimates for each Main round, selected "
            "workflow, and actual error exposure group. Points are not connected "
            "because workflow-group membership could change between Main rounds. "
            "Error bars show approximate 95% confidence intervals for cells with "
            "at least three observations."
        ),
    )
