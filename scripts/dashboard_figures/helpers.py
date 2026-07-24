"""Stable facade for reusable dashboard helper functions.

Implementations are grouped by responsibility under ``helper_modules``.
"""

from scripts.dashboard_figures.helper_modules.labels import (
    evaluator_color,
    evaluator_display_name,
    exposure_display_name,
    main_round_display_name,
    main_round_tick_labels,
    round_display_name,
    workflow_display_name,
)
from scripts.dashboard_figures.helper_modules.metrics import (
    add_passed_numeric,
    parse_requirement_results,
    pass_summary,
    quality_summary,
    wilson_interval,
)
from scripts.dashboard_figures.helper_modules.rankings import (
    build_valid_ranking_rows,
    normalize_ranking,
    parse_workflow_ranking,
    ranking_summary,
)
from scripts.dashboard_figures.helper_modules.rounds import (
    annotate_injected_error_round,
    drop_duplicate_participant_rounds,
    get_complete_main_round_participants,
    get_main_round_position,
    get_main_rounds,
    is_ai_supported_row,
    ordered_exposure_groups,
    phase_data,
    prepare_round_data,
    shade_main_rounds,
    shade_main_rounds_for_bar_axis,
)

__all__ = [
    "add_passed_numeric",
    "annotate_injected_error_round",
    "build_valid_ranking_rows",
    "drop_duplicate_participant_rounds",
    "evaluator_color",
    "evaluator_display_name",
    "exposure_display_name",
    "get_complete_main_round_participants",
    "get_main_round_position",
    "get_main_rounds",
    "is_ai_supported_row",
    "main_round_display_name",
    "main_round_tick_labels",
    "normalize_ranking",
    "ordered_exposure_groups",
    "parse_requirement_results",
    "parse_workflow_ranking",
    "pass_summary",
    "phase_data",
    "prepare_round_data",
    "quality_summary",
    "ranking_summary",
    "round_display_name",
    "shade_main_rounds",
    "shade_main_rounds_for_bar_axis",
    "wilson_interval",
    "workflow_display_name",
]
