"""Generate descriptive practice-round workflow-efficiency figures."""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.efficiency_analysis import (
    plot_completion_time_by_workflow_practice_rounds,
    plot_quality_time_efficiency_profile_practice_rounds,
)
from scripts.dashboard_figures.efficiency_analysis.common import (
    _prepare_efficiency_data,
)


def plot_efficiency(df: pd.DataFrame) -> None:
    """Generate descriptive practice-round workflow-efficiency figures."""
    practice_df, time_source = _prepare_efficiency_data(df)
    if practice_df.empty or time_source is None:
        return

    plot_completion_time_by_workflow_practice_rounds(practice_df, time_source)
    plot_quality_time_efficiency_profile_practice_rounds(practice_df, time_source)


__all__ = [
    "plot_completion_time_by_workflow_practice_rounds",
    "plot_efficiency",
    "plot_quality_time_efficiency_profile_practice_rounds",
]
