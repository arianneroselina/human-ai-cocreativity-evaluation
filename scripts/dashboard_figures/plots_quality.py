"""Generate output-quality figures."""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.helpers import phase_data
from scripts.dashboard_figures.quality_analysis import (
    plot_mixed_vs_solo_quality_practice_rounds,
    plot_mixed_workflow_direction_quality_practice_rounds,
    plot_overall_quality_by_workflow_practice_rounds,
    plot_quality_examples,
    plot_rating_dimensions_by_workflow_practice_rounds,
)
from scripts.dashboard_figures.quality_analysis.common import _prepare_quality_data


def plot_quality(dataframe: pd.DataFrame) -> None:
    """Generate practice-round output-quality figures."""
    prepared = _prepare_quality_data(dataframe)
    if prepared.empty:
        return

    practice_df = phase_data(prepared, "practice")
    if not practice_df.empty:
        plot_overall_quality_by_workflow_practice_rounds(practice_df)
        plot_rating_dimensions_by_workflow_practice_rounds(practice_df)
        plot_mixed_workflow_direction_quality_practice_rounds(practice_df)
        plot_mixed_vs_solo_quality_practice_rounds(practice_df)

    plot_quality_examples(prepared)
