"""Generate participant subjective-experience figures."""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.experience_analysis import (
    plot_ai_experience_by_practice_round_and_workflow,
    plot_satisfaction_by_practice_round_and_workflow,
    plot_satisfaction_vs_external_quality,
    plot_tlx_score_by_workflow_in_practice_rounds,
)
from scripts.dashboard_figures.experience_analysis.common import (
    _prepare_experience_data,
)
from scripts.dashboard_figures.helpers import phase_data


def plot_experience(dataframe: pd.DataFrame) -> None:
    """Generate participant subjective-experience figures."""
    prepared = _prepare_experience_data(dataframe)
    if prepared.empty:
        return

    practice_df = phase_data(prepared, "practice")
    if not practice_df.empty:
        plot_satisfaction_by_practice_round_and_workflow(practice_df)
        plot_ai_experience_by_practice_round_and_workflow(practice_df)
        plot_tlx_score_by_workflow_in_practice_rounds(practice_df)

    plot_satisfaction_vs_external_quality(prepared)
