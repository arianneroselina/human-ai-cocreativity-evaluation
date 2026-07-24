"""Round filtering, deduplication, and plot annotations."""

from __future__ import annotations

import pandas as pd

from scripts.config import (
    AI_SUPPORTED_WORKFLOWS,
    INJECTED_ERROR_LABEL,
    MAIN_ROUND_INDICES,
    WORKFLOW_ORDER,
)
from scripts.utils import parse_bool, require_columns


def get_main_rounds(df):
    return df[df["roundIndex"].isin(MAIN_ROUND_INDICES)].copy()


def get_main_round_position(round_index):
    mapping = {
        5: 1,
        6: 2,
        7: 3,
    }
    return mapping.get(round_index)


def get_complete_main_round_participants(df):
    participant_rounds = df.groupby("participantId")["roundIndex"].apply(set)

    complete_ids = participant_rounds[
        participant_rounds.apply(
            lambda rounds: set(MAIN_ROUND_INDICES).issubset(rounds)
        )
    ].index

    return df[df["participantId"].isin(complete_ids)].copy()


def drop_duplicate_participant_rounds(
    df: pd.DataFrame,
    participant_column: str = "participantId",
    round_column: str = "roundIndex",
) -> pd.DataFrame:
    """Keep the first row for each participant-round combination."""
    required_columns = {participant_column, round_column}

    if not required_columns.issubset(df.columns):
        return df

    return df.drop_duplicates(
        subset=[participant_column, round_column],
        keep="first",
    )


def prepare_round_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and normalise valid participant-round workflow records."""
    required = {"participantId", "roundIndex", "workflow"}

    if df.empty or not require_columns(df, required, "round-level data"):
        return pd.DataFrame(columns=sorted(required))

    prepared = drop_duplicate_participant_rounds(df.copy())
    prepared["roundIndex"] = pd.to_numeric(
        prepared["roundIndex"],
        errors="coerce",
    )
    prepared = prepared.dropna(
        subset=["participantId", "roundIndex", "workflow"],
    )
    prepared["roundIndex"] = prepared["roundIndex"].astype(int)

    return prepared[prepared["workflow"].isin(WORKFLOW_ORDER)].copy()


def phase_data(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Return observations belonging to the requested phase."""
    if phase not in {"practice", "main"}:
        raise ValueError("phase must be either 'practice' or 'main'")

    if "phase" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    return df.loc[df["phase"].eq(phase)].copy()


def ordered_exposure_groups(dataframe: pd.DataFrame) -> list[bool]:
    """Return available error-exposure values in a consistent order."""
    if "errorExposed" not in dataframe.columns:
        return []

    available = set(dataframe["errorExposed"].dropna())
    return [exposed for exposed in [True, False] if exposed in available]


def shade_main_rounds(
    ax,
    start_round=5,
    end_round=7,
    label="Main rounds (5–7)",
    color="#f2f2f2",
    alpha=0.8,
    label_y=0.03,
):
    """
    Adds a light background shade for the main rounds.

    Use this for line plots where the x-axis is the actual round number.
    Example: rounds 1, 2, 3, 4, 5, 6, 7.
    """
    left = start_round - 0.5
    right = end_round + 0.5
    center = (start_round + end_round) / 2

    ax.axvspan(
        left,
        right,
        color=color,
        alpha=alpha,
        zorder=0,
    )

    if label:
        ax.text(
            center,
            label_y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.75,
            },
        )


def shade_main_rounds_for_bar_axis(
    ax,
    round_indices,
    start_round=5,
    end_round=7,
    label="Main rounds (5–7)",
    color="#f2f2f2",
    alpha=0.8,
    label_y=0.03,
):
    """
    Adds background shading for bar plots where rounds are shown as categorical bars.

    Pandas bar plots use positions 0, 1, 2, ... instead of the actual round numbers.
    """
    round_positions = {
        int(round_index): position for position, round_index in enumerate(round_indices)
    }

    positions = [
        position
        for round_index, position in round_positions.items()
        if start_round <= round_index <= end_round
    ]

    if not positions:
        return

    left = min(positions) - 0.5
    right = max(positions) + 0.5
    center = (left + right) / 2

    ax.axvspan(
        left,
        right,
        color=color,
        alpha=alpha,
        zorder=0,
    )

    if label:
        ax.text(
            center,
            label_y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.75,
            },
        )


def is_ai_supported_row(row):
    if "isAiSupportedWorkflow" in row.index:
        return parse_bool(row["isAiSupportedWorkflow"])

    return row.get("workflow") in AI_SUPPORTED_WORKFLOWS
