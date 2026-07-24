"""Shared plot for binary Main-round rates by exposure and workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    MAIN_ROUND_INDICES,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    round_tick_labels,
    ordered_exposure_groups,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    INJECTED_ERROR_SPAN_COLOR,
    apply_standard_axes_style,
    INJECTED_ERROR_LABEL_COLOR,
)
from scripts.utils import save_figure, save_table


@dataclass(frozen=True)
class ExposureRatePlotConfig:
    """Text and column configuration for one binary-rate figure."""

    slug: str
    rate_column: str
    event_count_column: str
    total_count_column: str
    y_label: str
    title: str
    description: str
    figure_height: float = 5.6
    layout_bottom: float = 0.095


def _is_exposed(value: object) -> bool:
    """Interpret the configured exposure-group values consistently."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower().replace("_", "-")
    return normalized in {
        "true",
        "1",
        "yes",
        "exposed",
        "error-exposed",
    }


def plot_exposure_workflow_rate(
    summary: pd.DataFrame,
    config: ExposureRatePlotConfig,
) -> None:
    """Plot separate binary-rate estimates for every exposure/round/workflow cell.

    Points are deliberately not connected because participants could switch
    workflows between Main rounds. Confidence intervals are expected to be
    Wilson intervals in percentage points.
    """
    if summary.empty:
        return

    required_columns = {
        "errorExposed",
        "roundIndex",
        "workflow",
        config.rate_column,
        config.event_count_column,
        config.total_count_column,
        "lowerCI",
        "upperCI",
    }
    missing = required_columns - set(summary.columns)
    if missing:
        raise ValueError(
            f"Cannot plot {config.slug}; summary is missing {sorted(missing)}"
        )

    plot_summary = summary.copy()
    plot_summary["workflowLabel"] = plot_summary["workflow"].map(workflow_display_name)
    plot_summary["exposureLabel"] = plot_summary["errorExposed"].map(
        exposure_display_name
    )
    save_table(plot_summary, config.slug, index=False)

    groups = ordered_exposure_groups(plot_summary)
    observed_rounds = set(
        pd.to_numeric(plot_summary["roundIndex"], errors="coerce").dropna().astype(int)
    )
    main_rounds = [
        round_index
        for round_index in MAIN_ROUND_INDICES
        if round_index in observed_rounds
    ]
    if not groups or not main_rounds:
        return

    workflow_offsets = dict(
        zip(WORKFLOW_ORDER, np.linspace(-0.18, 0.18, len(WORKFLOW_ORDER)))
    )

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.7 * len(groups), config.figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    for axis, group in zip(axes, groups):
        group_summary = plot_summary.loc[plot_summary["errorExposed"].eq(group)]

        if _is_exposed(group):
            add_injected_error_round_marker(axis, main_rounds)

        for workflow in WORKFLOW_ORDER:
            workflow_summary = (
                group_summary.loc[group_summary["workflow"].eq(workflow)]
                .set_index("roundIndex")
                .reindex(main_rounds)
            )

            rates = pd.to_numeric(
                workflow_summary[config.rate_column], errors="coerce"
            ).to_numpy(dtype=float)
            lower_ci = pd.to_numeric(
                workflow_summary["lowerCI"], errors="coerce"
            ).to_numpy(dtype=float)
            upper_ci = pd.to_numeric(
                workflow_summary["upperCI"], errors="coerce"
            ).to_numpy(dtype=float)
            event_counts = (
                pd.to_numeric(
                    workflow_summary[config.event_count_column],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )
            total_counts = (
                pd.to_numeric(
                    workflow_summary[config.total_count_column],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(dtype=int)
            )

            valid = np.isfinite(rates)
            if not valid.any():
                continue

            x_values = np.asarray(main_rounds, dtype=float) + workflow_offsets[workflow]
            valid_ci = valid & np.isfinite(lower_ci) & np.isfinite(upper_ci)

            if valid_ci.any():
                clipped_lower = np.clip(lower_ci[valid_ci], 0, 100)
                clipped_upper = np.clip(upper_ci[valid_ci], 0, 100)
                axis.errorbar(
                    x_values[valid_ci],
                    rates[valid_ci],
                    yerr=np.vstack(
                        [
                            rates[valid_ci] - clipped_lower,
                            clipped_upper - rates[valid_ci],
                        ]
                    ),
                    fmt="none",
                    ecolor=WORKFLOW_COLORS[workflow],
                    elinewidth=1.25,
                    capsize=4,
                    capthick=1.1,
                    alpha=0.9,
                    zorder=2,
                )

            axis.scatter(
                x_values[valid],
                rates[valid],
                s=66,
                color=WORKFLOW_COLORS[workflow],
                edgecolor="white",
                linewidth=0.9,
                label=workflow_display_name(workflow),
                zorder=3,
            )

            for x_value, rate, events, total in zip(
                x_values[valid],
                rates[valid],
                event_counts[valid],
                total_counts[valid],
            ):
                if rate >= 82:
                    text_offset = (0, -11)
                    vertical_alignment = "top"
                else:
                    text_offset = (0, 8)
                    vertical_alignment = "bottom"

                axis.annotate(
                    f"{events}/{total}",
                    (x_value, rate),
                    xytext=text_offset,
                    textcoords="offset points",
                    ha="center",
                    va=vertical_alignment,
                    fontsize=7,
                    color=WORKFLOW_COLORS[workflow],
                    zorder=4,
                )

        axis.set_title(
            exposure_display_name(group),
            fontsize=11.5,
            pad=34,
        )
        axis.set_xticks(main_rounds)
        axis.set_xticklabels(round_tick_labels(main_rounds))
        axis.set_xlim(min(main_rounds) - 0.45, max(main_rounds) + 0.45)
        axis.set_ylim(-4, 108)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_xlabel("Main round")
        apply_standard_axes_style(axis, grid_axis="y")

    axes[0].set_ylabel(config.y_label)

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

    fig.suptitle(config.title, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, config.layout_bottom, 0.84, 0.94))

    save_figure(
        fig,
        config.slug,
        config.title,
        config.description,
    )


from collections.abc import Sequence

from matplotlib.axes import Axes


def add_injected_error_round_marker(
    axis: Axes,
    rounds: Sequence[int],
) -> None:
    """Highlight the Main round containing the injected AI error."""
    if INJECTED_ERROR_ROUND_INDEX not in rounds:
        return

    axis.axvspan(
        INJECTED_ERROR_ROUND_INDEX - 0.32,
        INJECTED_ERROR_ROUND_INDEX + 0.32,
        facecolor=INJECTED_ERROR_SPAN_COLOR,
        edgecolor="none",
        zorder=0,
    )

    axis.text(
        INJECTED_ERROR_ROUND_INDEX,
        0.975,
        "Injected AI error",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=INJECTED_ERROR_LABEL_COLOR,
        fontstyle="italic",
        zorder=5,
    )
