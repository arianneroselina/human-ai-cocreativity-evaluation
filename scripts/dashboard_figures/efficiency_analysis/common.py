"""Shared preparation and summaries for workflow-efficiency figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import QUALITY_PRIMARY_METRIC, WORKFLOW_ORDER
from scripts.dashboard_figures.helpers import (
    phase_data,
    workflow_display_name,
)
from scripts.dashboard_figures.summaries import numeric_summary
from scripts.utils import require_columns


def _add_total_completion_time(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | tuple[pd.DataFrame, None]:
    """Add total completion time in minutes, including pauses where available.

    The helper prefers explicit elapsed or total-duration columns. It never uses
    ``effectiveTimeMinutes`` alone because that measure may exclude pauses.
    """
    prepared = dataframe.copy()

    if "totalCompletionTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["totalCompletionTimeMinutes"],
            errors="coerce",
        )
        return prepared, "totalCompletionTimeMinutes"

    if "elapsedTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["elapsedTimeMinutes"],
            errors="coerce",
        )
        return prepared, "elapsedTimeMinutes"

    if "totalTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["totalTimeMinutes"],
            errors="coerce",
        )
        return prepared, "totalTimeMinutes"

    if {"effectiveTimeMinutes", "pauseTimeMinutes"}.issubset(prepared.columns):
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["effectiveTimeMinutes"], errors="coerce"
        ) + pd.to_numeric(prepared["pauseTimeMinutes"], errors="coerce")
        return prepared, "effectiveTimeMinutes + pauseTimeMinutes"

    if {"effectiveTimeMs", "pauseTimeMs"}.issubset(prepared.columns):
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["effectiveTimeMs"], errors="coerce")
            + pd.to_numeric(prepared["pauseTimeMs"], errors="coerce")
        ) / 60000
        return prepared, "effectiveTimeMs + pauseTimeMs"

    if "elapsedTimeMs" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["elapsedTimeMs"], errors="coerce") / 60000
        )
        return prepared, "elapsedTimeMs"

    if {"startedAt", "submittedAt"}.issubset(prepared.columns):
        started = pd.to_datetime(prepared["startedAt"], errors="coerce", utc=True)
        submitted = pd.to_datetime(
            prepared["submittedAt"],
            errors="coerce",
            utc=True,
        )
        prepared["totalCompletionTimeMinutes"] = (
            submitted - started
        ).dt.total_seconds() / 60
        return prepared, "submittedAt - startedAt"

    if "timeMs" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["timeMs"], errors="coerce") / 60000
        )
        return prepared, "timeMs"

    print(
        "Skipping efficiency figures; no total or elapsed completion-time "
        "measure was found. Effective time alone is not used because pauses "
        "must be included."
    )
    return pd.DataFrame(), None


def _prepare_efficiency_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Return valid practice-round quality and total-time observations."""
    required = {"workflow", QUALITY_PRIMARY_METRIC}
    if not require_columns(df, required, "practice-round efficiency"):
        return pd.DataFrame(), None

    practice_df = phase_data(df, "practice")
    if practice_df.empty:
        return pd.DataFrame(), None

    practice_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        practice_df[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    practice_df, time_source = _add_total_completion_time(practice_df)
    if practice_df.empty:
        return pd.DataFrame(), None

    practice_df = practice_df.dropna(
        subset=[
            "workflow",
            QUALITY_PRIMARY_METRIC,
            "totalCompletionTimeMinutes",
        ]
    )
    practice_df = practice_df.loc[
        practice_df["workflow"].isin(WORKFLOW_ORDER)
        & practice_df["totalCompletionTimeMinutes"].gt(0)
    ].copy()

    return practice_df, time_source


def _workflow_order_present(dataframe: pd.DataFrame) -> list[str]:
    """Return available workflows in canonical display order."""
    available = set(dataframe["workflow"].dropna().unique())
    return [workflow for workflow in WORKFLOW_ORDER if workflow in available]


def _workflow_efficiency_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise completion time and quality for each workflow."""
    rows = []

    for workflow in _workflow_order_present(dataframe):
        workflow_df = dataframe.loc[dataframe["workflow"].eq(workflow)]
        time_summary = numeric_summary(workflow_df["totalCompletionTimeMinutes"])
        quality_summary = numeric_summary(workflow_df[QUALITY_PRIMARY_METRIC])

        rows.append(
            {
                "workflow": workflow,
                "workflowLabel": workflow_display_name(workflow),
                "count": min(
                    int(time_summary["count"]),
                    int(quality_summary["count"]),
                ),
                "meanCompletionTimeMinutes": time_summary["mean"],
                "medianCompletionTimeMinutes": time_summary["median"],
                "completionTimeStd": time_summary["std"],
                "completionTimeCiLow": max(
                    0.0,
                    float(time_summary["ciLow"]),
                )
                if pd.notna(time_summary["ciLow"])
                else np.nan,
                "completionTimeCiHigh": time_summary["ciHigh"],
                "meanQuality": quality_summary["mean"],
                "medianQuality": quality_summary["median"],
                "qualityStd": quality_summary["std"],
                "qualityCiLow": quality_summary["ciLow"],
                "qualityCiHigh": quality_summary["ciHigh"],
            }
        )

    return pd.DataFrame(rows)
