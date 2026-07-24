"""Reusable plotting primitives for round-based dashboard figures."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.dashboard_figures.style import WORKFLOW_COLORS


def plot_workflow_round_series(
    ax,
    summary: pd.DataFrame,
    workflow: str,
    rounds: Sequence[int],
    metric: str,
    point_offset: float = 0.0,
    x_column: str = "roundIndex",
    *,
    connect_points: bool = True,
) -> None:
    """Plot one workflow mean series with confidence intervals.

    Missing workflow-round cells remain gaps. This avoids visually implying an
    observation or transition where no data exists.
    """
    workflow_summary = (
        summary.loc[summary["workflow"].eq(workflow) & summary["metric"].eq(metric)]
        .set_index(x_column)
        .reindex(rounds)
    )

    valid_means = workflow_summary["mean"].notna().to_numpy()
    if not valid_means.any():
        return

    x_values = np.asarray(rounds, dtype=float) + point_offset
    means = workflow_summary["mean"].to_numpy(dtype=float)
    color = WORKFLOW_COLORS[workflow]

    ax.plot(
        x_values[valid_means],
        means[valid_means],
        marker="o",
        linestyle="-" if connect_points else "none",
        linewidth=1.8 if connect_points else 0,
        markersize=5.5,
        color=color,
        label=workflow_display_name(workflow),
        zorder=3,
    )

    valid_intervals = (
        workflow_summary["lowerCI"].notna()
        & workflow_summary["upperCI"].notna()
        & workflow_summary["mean"].notna()
    ).to_numpy()

    if not valid_intervals.any():
        return

    lower_ci = workflow_summary["lowerCI"].to_numpy(dtype=float)
    upper_ci = workflow_summary["upperCI"].to_numpy(dtype=float)

    lower_errors = means[valid_intervals] - lower_ci[valid_intervals]
    upper_errors = upper_ci[valid_intervals] - means[valid_intervals]

    ax.errorbar(
        x_values[valid_intervals],
        means[valid_intervals],
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor=color,
        capsize=3,
        elinewidth=1.0,
        capthick=1.0,
        alpha=0.9,
        zorder=2,
    )
