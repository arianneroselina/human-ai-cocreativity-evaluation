"""Workflow-selection analysis modules."""

from scripts.dashboard_figures.workflow_analysis.preference import (
    plot_final_workflow_preference,
    plot_stated_vs_observed_workflow_behaviour,
)
from scripts.dashboard_figures.workflow_analysis.switching import (
    plot_workflow_switching_behaviour,
)
from scripts.dashboard_figures.workflow_analysis.transitions import (
    plot_main_workflow_transitions,
    plot_practice_to_first_choice_transition,
    plot_workflow_retention,
)
from scripts.dashboard_figures.workflow_analysis.usage import (
    plot_first_voluntary_workflow_choice,
    plot_participant_workflow_trajectories,
    plot_total_workflow_usage_counts,
    plot_workflow_distribution,
)

__all__ = [
    "plot_final_workflow_preference",
    "plot_first_voluntary_workflow_choice",
    "plot_main_workflow_transitions",
    "plot_participant_workflow_trajectories",
    "plot_practice_to_first_choice_transition",
    "plot_stated_vs_observed_workflow_behaviour",
    "plot_total_workflow_usage_counts",
    "plot_workflow_distribution",
    "plot_workflow_retention",
    "plot_workflow_switching_behaviour",
]
