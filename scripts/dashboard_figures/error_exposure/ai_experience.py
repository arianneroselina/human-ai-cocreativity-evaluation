"""Main-round AI interaction ratings by exposure and workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    AI_EXPERIENCE_METRICS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.error_exposure.rate_plot import (
    add_injected_error_round_marker,
    _is_exposed,
)
from scripts.dashboard_figures.helper_modules.labels import round_display_name
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    ordered_exposure_groups,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    apply_standard_axes_style, VALUE_LABEL_FONT_SIZE, SUBTITLE_FONT_SIZE, INSIDE_LABEL_FONT_SIZE,
)
from scripts.dashboard_figures.summaries import grouped_metric_summary
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


def plot_main_round_ai_experience_by_exposure(
    main_df: pd.DataFrame,
) -> None:
    """Show AI interaction ratings by Main round, exposure, and workflow.

    Each point represents a separate Main round and workflow subgroup.
    Points are not connected because participants could switch workflows
    between Main rounds.
    """
    slug = "103_main_round_ai_experience_by_exposure_and_workflow"

    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
    }
    if not require_columns(
        main_df,
        required,
        "AI interaction ratings in Main rounds",
    ):
        return

    available_metrics = [
        metric
        for metric in AI_EXPERIENCE_METRICS
        if metric in main_df.columns
        and pd.to_numeric(
            main_df[metric],
            errors="coerce",
        )
        .notna()
        .any()
    ]
    if not available_metrics:
        return

    ai_workflows = [workflow for workflow in WORKFLOW_ORDER if workflow != "human"]

    plot_df = main_df.loc[main_df["workflow"].isin(ai_workflows)].copy()

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(
            plot_df[metric],
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
        metric_columns=available_metrics,
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    summary["metricLabel"] = summary["metric"].map(AI_EXPERIENCE_METRICS)

    save_table(summary, slug, index=False)

    # Separate the three workflow estimates around each round position.
    workflow_offsets = dict(
        zip(
            ai_workflows,
            np.linspace(-0.14, 0.14, len(ai_workflows)),
        )
    )

    fig, axes = plt.subplots(
        len(available_metrics),
        len(groups),
        figsize=(
            6.3 * len(groups),
            3.55 * len(available_metrics),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    round_labels = [
        round_display_name(round_index) for index, round_index in enumerate(rounds)
    ]

    for row_index, metric in enumerate(available_metrics):
        for column_index, group in enumerate(groups):
            axis = axes[row_index, column_index]

            panel_summary = summary.loc[
                summary["errorExposed"].eq(group) & summary["metric"].eq(metric)
            ]

            for workflow in ai_workflows:
                workflow_summary = (
                    panel_summary.loc[panel_summary["workflow"].eq(workflow)]
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

                # Separate points without lines between rounds.
                axis.scatter(
                    x_values[valid],
                    means[valid],
                    s=46,
                    color=WORKFLOW_COLORS[workflow],
                    edgecolor="white",
                    linewidth=0.7,
                    label=workflow_display_name(workflow),
                    zorder=4,
                )

                # Show confidence intervals only for cells with at least
                # three observations.
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

                    lower_ci = ci_rows["lowerCI"].to_numpy(dtype=float)
                    upper_ci = ci_rows["upperCI"].to_numpy(dtype=float)

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
                    if mean >= 4.4:
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

            if _is_exposed(group):
                add_injected_error_round_marker(axis, rounds)

            axis.set_xticks(rounds)
            axis.set_xticklabels(round_labels)
            axis.set_ylim(0.7, 5.42)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_xlabel("Main round")

            if column_index == 0:
                axis.set_ylabel(f"{AI_EXPERIENCE_METRICS[metric]}\nMean rating (1–5)")

            if row_index == 0:
                axis.set_title(exposure_display_name(group))

            apply_standard_axes_style(
                axis,
                grid_axis="y",
            )

    # Collect unique legend entries from all panels.
    legend_items = {}

    for axis in axes.flatten():
        handles, labels = axis.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)

    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            title="AI supported workflow",
            bbox_to_anchor=(0.87, 0.5),
            loc="center left",
        )

    fig.suptitle(
        "AI Interaction Ratings by Main Round, Workflow, and Error Exposure",
        y=0.995,
    )

    fig.tight_layout(rect=(0, 0.045, 0.84, 0.97))

    save_figure(
        fig,
        slug,
        "AI Interaction Ratings by Main Round, Workflow, and Error Exposure",
        (
            "Separate mean AI interaction ratings for each Main round, "
            "AI supported workflow, and actual error exposure group. Points "
            "are not connected because participants could switch workflows "
            "between Main rounds. Error bars show approximate 95% confidence "
            "intervals for cells with at least three observations."
        ),
    )
