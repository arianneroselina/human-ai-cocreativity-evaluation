"""Injected AI-error exposure figure modules."""

from scripts.dashboard_figures.error_exposure.ai_experience import (
    plot_main_round_ai_experience_by_exposure,
)
from scripts.dashboard_figures.error_exposure.choice import (
    plot_main_round1_workflow_choice,
)
from scripts.dashboard_figures.error_exposure.constraints import (
    plot_main_constraint_fulfillment_by_exposure,
)
from scripts.dashboard_figures.error_exposure.interviews import (
    plot_injected_error_awareness,
    plot_other_ai_error_types,
)
from scripts.dashboard_figures.error_exposure.line_count import (
    plot_main_line_count_error_by_exposure,
)
from scripts.dashboard_figures.error_exposure.post_error_choice import (
    plot_post_error_workflow_choices_by_exposure,
)
from scripts.dashboard_figures.error_exposure.preference import (
    plot_final_workflow_preference_by_reported_ai_errors,
)
from scripts.dashboard_figures.error_exposure.quality import (
    plot_main_round_quality_by_error_exposure,
)
from scripts.dashboard_figures.error_exposure.satisfaction import (
    plot_main_round_satisfaction_by_exposure,
)
from scripts.dashboard_figures.error_exposure.workload import (
    plot_main_round_tlx_by_exposure_and_workflow,
)

__all__ = [
    "plot_final_workflow_preference_by_reported_ai_errors",
    "plot_injected_error_awareness",
    "plot_main_constraint_fulfillment_by_exposure",
    "plot_main_line_count_error_by_exposure",
    "plot_main_round1_workflow_choice",
    "plot_main_round_ai_experience_by_exposure",
    "plot_main_round_quality_by_error_exposure",
    "plot_main_round_satisfaction_by_exposure",
    "plot_main_round_tlx_by_exposure_and_workflow",
    "plot_other_ai_error_types",
    "plot_post_error_workflow_choices_by_exposure",
]
