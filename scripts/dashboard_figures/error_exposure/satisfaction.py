"""Main-round satisfaction by injected-error exposure and workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    SATISFACTION_COLUMN,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    main_round_tick_labels,
    ordered_exposure_groups,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    FOOTNOTE_TEXT_COLOR,
    apply_standard_axes_style,
)
from scripts.dashboard_figures.summaries import grouped_metric_summary
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


def plot_main_round_satisfaction_by_exposure(
    main_df: pd.DataFrame,
) -> None:
    """Show Main round satisfaction by exposure group and selected workflow.

    Each point represents a separate round-workflow mean. Points are not
    connected because the participants selecting a workflow can change between
    Main rounds.
    """
    slug = "102_main_round_satisfaction_by_exposure_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        SATISFACTION_COLUMN,
    }
    if not require_columns(
        main_df,
        required,
        "Main round satisfaction by error exposure",
    ):
        return

    plot_df = main_df.dropna(
        subset=[
            "roundIndex",
            "workflow",
            "errorExposed",
            SATISFACTION_COLUMN,
        ]
    ).copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )
    plot_df[SATISFACTION_COLUMN] = pd.to_numeric(
        plot_df[SATISFACTION_COLUMN],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["roundIndex", SATISFACTION_COLUMN])
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
        metric_columns=[SATISFACTION_COLUMN],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    save_table(summary, slug, index=False)

    # Slightly separate workflow estimates around each Main round position.
    workflow_offsets = dict(
        zip(
            WORKFLOW_ORDER,
            np.linspace(-0.18, 0.18, len(WORKFLOW_ORDER)),
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
        group_summary = summary.loc[
            summary["errorExposed"].eq(group)
            & summary["metric"].eq(SATISFACTION_COLUMN)
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

            # Separate points: no connecting lines.
            axis.scatter(
                x_values[valid],
                means[valid],
                s=48,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.7,
                label=workflow_display_name(workflow),
                zorder=4,
            )

            ci_valid = (
                workflow_summary["mean"].notna()
                & workflow_summary["lowerCI"].notna()
                & workflow_summary["upperCI"].notna()
            ).to_numpy()

            if ci_valid.any():
                lower_ci = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna(),
                    "lowerCI",
                ].to_numpy(dtype=float)

                upper_ci = workflow_summary.loc[
                    workflow_summary["mean"].notna()
                    & workflow_summary["lowerCI"].notna()
                    & workflow_summary["upperCI"].notna(),
                    "upperCI",
                ].to_numpy(dtype=float)

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
                    elinewidth=1.1,
                    capsize=3,
                    alpha=0.9,
                    zorder=3,
                )

            valid_rows = workflow_summary.loc[workflow_summary["mean"].notna()]

            for x_value, y_value, count in zip(
                x_values[valid],
                means[valid],
                valid_rows["count"].to_numpy(dtype=int),
            ):
                axis.annotate(
                    f"n={count}",
                    (x_value, y_value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                )

        axis.axvline(
            INJECTED_ERROR_ROUND_INDEX,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.55,
            zorder=1,
        )

        axis.set_title(exposure_display_name(group))
        axis.set_xticks(rounds)
        axis.set_xticklabels(main_round_tick_labels(rounds))
        axis.set_xlabel("Main round")
        axis.set_ylim(0.7, 5.42)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))

        apply_standard_axes_style(axis, grid_axis="y")

    axes[0].set_ylabel("Satisfaction rating (1–5)")

    # Collect all workflow labels that appear in either exposure panel.
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
            bbox_to_anchor=(0.99, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "Participant Satisfaction by Main Round, Workflow, and Error Exposure",
        fontsize=13,
        y=0.99,
    )

    fig.text(
        0.01,
        0.01,
        (
            "Points show separate round-workflow means and are not connected "
            "because participants could switch workflows between rounds. "
            "The dashed line marks the injected-error round. Cells with very "
            "small sample sizes should be interpreted cautiously."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color=FOOTNOTE_TEXT_COLOR,
    )

    fig.tight_layout(rect=(0, 0.06, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Participant Satisfaction by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean satisfaction estimates for each Main round workflow "
            "cell, shown by actual error-exposure group. Points are not connected "
            "because workflow-group membership could change between rounds. "
            "Error bars show approximate 95% confidence intervals; cells with one "
            "observation have no confidence interval."
        ),
    )
