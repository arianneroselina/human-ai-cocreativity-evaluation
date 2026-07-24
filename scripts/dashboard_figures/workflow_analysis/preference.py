"""Compatibility exports for workflow-preference figures."""

from scripts.dashboard_figures.workflow_analysis.final_preference import (
    plot_final_workflow_preference,
)
from scripts.dashboard_figures.workflow_analysis.observed_behavior import (
    plot_stated_vs_observed_workflow_behaviour,
)

__all__ = [
    "plot_final_workflow_preference",
    "plot_stated_vs_observed_workflow_behaviour",
]
