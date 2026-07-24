"""Workflow-efficiency figure implementations."""

from scripts.dashboard_figures.efficiency_analysis.completion_time import (
    plot_completion_time_by_workflow_practice_rounds,
)
from scripts.dashboard_figures.efficiency_analysis.quality_time import (
    plot_quality_time_efficiency_profile_practice_rounds,
)

__all__ = [
    "plot_completion_time_by_workflow_practice_rounds",
    "plot_quality_time_efficiency_profile_practice_rounds",
]
