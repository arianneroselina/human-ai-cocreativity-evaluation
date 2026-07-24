"""Shared preparation and categories for constraint figures."""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.helpers import (
    add_passed_numeric,
    parse_requirement_results,
)
from scripts.utils import parse_bool_or_none


CONSTRAINT_TYPE_ORDER = [
    "Line count",
    "Required words",
    "Forbidden words",
    "Total word count",
    "Words per line",
    "Special format",
]


CONSTRAINT_TYPE_CATEGORIES = {
    "lines-": "Line count",
    "must-": "Required words",
    "avoid-": "Forbidden words",
    "maxwords-": "Total word count",
    "maxwpl-": "Words per line",
}


FAILURE_PROFILE_ORDER = [
    "All constraints passed",
    "Line-count rule only",
    "One other rule only",
    "Multiple rules failed",
    "Failure details unavailable",
]


FAILURE_PROFILE_COLORS = {
    "All constraints passed": "#009E73",
    "Line-count rule only": "#E69F00",
    "One other rule only": "#56B4E9",
    "Multiple rules failed": "#D55E00",
    "Failure details unavailable": "#999999",
}


def prepare_constraint_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add variables required for constraint analysis."""
    prepared = add_passed_numeric(df)

    if prepared.empty:
        return prepared

    if "constraintScore" in prepared.columns:
        prepared["constraintScore"] = pd.to_numeric(
            prepared["constraintScore"],
            errors="coerce",
        ).clip(lower=0, upper=100)

    if "effectiveTimeMinutes" in prepared.columns:
        prepared["effectiveTimeMinutes"] = pd.to_numeric(
            prepared["effectiveTimeMinutes"],
            errors="coerce",
        )
    elif "timeMs" in prepared.columns:
        prepared["effectiveTimeMinutes"] = (
            pd.to_numeric(
                prepared["timeMs"],
                errors="coerce",
            )
            / 60_000
        )

    return prepared


def _constraint_type(rule_id: str) -> str:
    for prefix, label in CONSTRAINT_TYPE_CATEGORIES.items():
        if rule_id.startswith(prefix):
            return label
    return "Special format"


def _constraint_failure_profile(
    passed_numeric: float,
    requirement_results,
) -> str | None:
    """Classify each round into one mutually exclusive failure profile."""
    if pd.isna(passed_numeric):
        return None

    if passed_numeric == 1:
        return "All constraints passed"

    failed_rule_ids = []

    for item in parse_requirement_results(requirement_results):
        if parse_bool_or_none(item.get("passed")) is False:
            failed_rule_ids.append(str(item.get("id", "")))

    if not failed_rule_ids:
        return "Failure details unavailable"

    line_count_failures = sum(
        rule_id.startswith("lines-") for rule_id in failed_rule_ids
    )

    if len(failed_rule_ids) == 1:
        if line_count_failures == 1:
            return "Line-count rule only"

        return "One other rule only"

    return "Multiple rules failed"


def _explode_requirement_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create one row per individual requirement check."""
    rows = []

    for _, record in dataframe.iterrows():
        for item in parse_requirement_results(record.get("requirementResults")):
            passed = parse_bool_or_none(item.get("passed"))
            if passed is None:
                continue

            rule_id = str(item.get("id", ""))
            rows.append(
                {
                    "participantId": record.get("participantId"),
                    "roundIndex": record.get("roundIndex"),
                    "workflow": record.get("workflow"),
                    "constraintType": _constraint_type(rule_id),
                    "constraintLabel": item.get("label", rule_id),
                    "passed": passed,
                }
            )

    return pd.DataFrame(rows)
