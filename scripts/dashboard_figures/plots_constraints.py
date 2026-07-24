"""Generate practice-round constraint-fulfillment figures."""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.constraint_analysis import (
    plot_practice_constraint_failure_profile_by_workflow,
    plot_practice_constraint_pass_rate_by_workflow,
    plot_practice_failure_breakdown_by_constraint_type,
)
from scripts.dashboard_figures.constraint_analysis.common import (
    prepare_constraint_data,
)
from scripts.dashboard_figures.helpers import phase_data
from scripts.utils import require_columns


def plot_constraints(df: pd.DataFrame) -> None:
    """Generate practice-round constraint-fulfillment figures."""
    prepared = prepare_constraint_data(df)
    practice_df = phase_data(prepared, "practice")

    required = {"requirementResults", "workflow", "passedNumeric"}
    if practice_df.empty or not require_columns(
        practice_df,
        required,
        "practice-round constraint fulfillment",
    ):
        return

    plot_practice_constraint_pass_rate_by_workflow(practice_df)
    plot_practice_constraint_failure_profile_by_workflow(practice_df)
    plot_practice_failure_breakdown_by_constraint_type(practice_df)


__all__ = [
    "plot_constraints",
    "plot_practice_constraint_failure_profile_by_workflow",
    "plot_practice_constraint_pass_rate_by_workflow",
    "plot_practice_failure_breakdown_by_constraint_type",
]
