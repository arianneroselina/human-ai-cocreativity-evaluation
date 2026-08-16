"""Generate all injected AI-error exposure figures.

The individual figures live in ``dashboard_figures.error_exposure`` so each
module stays focused and can be tested or changed independently.
"""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.error_exposure import (
    plot_final_workflow_preference_by_reported_ai_errors,
    plot_injected_error_awareness,
    plot_main_constraint_fulfillment_by_exposure,
    plot_main_line_count_error_by_exposure,
    plot_main_round1_workflow_choice,
    plot_main_round_ai_experience_by_exposure,
    plot_main_round_quality_by_error_exposure,
    plot_main_round_satisfaction_by_exposure,
    plot_main_round_tlx_by_exposure_and_workflow,
    plot_other_ai_error_types,
    plot_post_error_workflow_choices_by_exposure,
    plot_post_error_workflow_choices_by_awareness,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.helpers import (
    add_passed_numeric,
    build_valid_ranking_rows,
    phase_data,
)
from scripts.utils import require_columns


def plot_error_exposure(
    prepared: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> None:
    """Generate injected-error exposure and Main-round outcome figures."""
    required = {
        "participantId",
        "roundIndex",
        "workflow",
        "errorExposed",
    }
    if prepared.empty or not require_columns(
        prepared,
        required,
        "error-exposure data",
    ):
        return

    prepared = add_passed_numeric(prepared)
    prepared["participantId"] = prepared["participantId"].astype(str)
    ranking_rows, _ = build_valid_ranking_rows(feedback_df)
    interview_notes = load_participant_interview_notes(prepared)

    main_df = phase_data(prepared, "main")
    if main_df.empty:
        return

    plot_main_round1_workflow_choice(prepared)
    plot_main_round_satisfaction_by_exposure(main_df)
    plot_main_round_ai_experience_by_exposure(main_df)
    plot_main_round_tlx_by_exposure_and_workflow(main_df)
    plot_post_error_workflow_choices_by_exposure(prepared)
    plot_post_error_workflow_choices_by_awareness(prepared, interview_notes)
    plot_main_round_quality_by_error_exposure(main_df)
    plot_main_constraint_fulfillment_by_exposure(main_df)
    plot_main_line_count_error_by_exposure(main_df)
    plot_final_workflow_preference_by_reported_ai_errors(
        ranking_rows,
        prepared,
        interview_notes,
    )
    plot_injected_error_awareness(prepared, interview_notes)
    plot_other_ai_error_types(prepared, interview_notes)
