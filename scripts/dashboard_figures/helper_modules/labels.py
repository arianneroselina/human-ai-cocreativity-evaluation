"""Canonical display labels and colors for dashboard figures."""

from __future__ import annotations

import hashlib

from scripts.config import (
    EVALUATOR_LABELS,
    EXPOSURE_LABELS,
    MAIN_ROUND_INDICES,
    ROUND_LABELS,
    WORKFLOW_LABELS,
)
from scripts.dashboard_figures.style import (
    EVALUATOR_COLORS,
    EVALUATOR_FALLBACK_COLORS,
)


def workflow_display_name(workflow):
    return WORKFLOW_LABELS.get(workflow, workflow)


def round_display_name(round_index):
    return ROUND_LABELS.get(round_index, f"Round {round_index}")


def round_tick_labels(round_indices) -> list[str]:
    """Return canonical labels for a sequence of Main-round indices."""
    return [round_display_name(round_index) for round_index in round_indices]


def exposure_display_name(exposed: bool) -> str:
    return EXPOSURE_LABELS.get(exposed, str(exposed))


def evaluator_display_name(evaluator_id: str) -> str:
    return EVALUATOR_LABELS.get(str(evaluator_id), str(evaluator_id))


def evaluator_color(evaluator_id: str) -> str:
    """Return a stable color for configured and unexpected evaluators."""
    normalized_id = str(evaluator_id)
    configured_color = EVALUATOR_COLORS.get(normalized_id)
    if configured_color is not None:
        return configured_color

    digest = hashlib.sha256(normalized_id.encode("utf-8")).digest()
    palette_index = int.from_bytes(digest[:2], "big") % len(EVALUATOR_FALLBACK_COLORS)
    return EVALUATOR_FALLBACK_COLORS[palette_index]
