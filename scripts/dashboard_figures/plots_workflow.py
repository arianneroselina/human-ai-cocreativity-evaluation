"""Generate workflow-selection figures.

Implementation is split across ``dashboard_figures.workflow_analysis`` to keep
usage, transitions, switching, and preference analyses independent.
"""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.helpers import (
    build_valid_ranking_rows,
    phase_data,
)
from scripts.dashboard_figures.workflow_analysis import (
    plot_final_workflow_preference,
    plot_first_voluntary_workflow_choice,
    plot_main_workflow_transitions,
    plot_participant_workflow_trajectories,
    plot_practice_to_first_choice_transition,
    plot_stated_vs_observed_workflow_behaviour,
    plot_total_workflow_usage_counts,
    plot_workflow_distribution,
    plot_workflow_retention,
    plot_workflow_switching_behaviour,
)


def plot_workflow(
    dataframe: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> None:
    """Generate the workflow-selection figures in analysis order."""
    main_df = phase_data(dataframe, "main")
    if main_df.empty:
        return

    ranking_rows, audit_df = build_valid_ranking_rows(feedback_df)

    plot_total_workflow_usage_counts(main_df)
    plot_first_voluntary_workflow_choice(main_df)
    plot_workflow_distribution(main_df)
    plot_participant_workflow_trajectories(main_df)
    plot_practice_to_first_choice_transition(dataframe)
    plot_main_workflow_transitions(main_df)
    plot_workflow_retention(main_df)
    plot_workflow_switching_behaviour(main_df)
    plot_final_workflow_preference(ranking_rows, audit_df)
    plot_stated_vs_observed_workflow_behaviour(ranking_rows, main_df)
