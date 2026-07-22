import json
import re

import numpy as np
import pandas as pd

from scripts.config import (
    MAIN_ROUND_INDICES,
    WORKFLOW_LABELS,
    ROUND_LABELS,
    EXPOSURE_LABELS,
    EVALUATOR_LABELS,
    WORKFLOW_ORDER,
    QUALITY_PRIMARY_METRIC,
    CI_Z_VALUE,
)
from scripts.utils import parse_bool, require_columns, parse_bool_or_none


def workflow_display_name(workflow):
    return WORKFLOW_LABELS.get(workflow, workflow)


def round_display_name(round_index):
    return ROUND_LABELS.get(round_index, f"Round {round_index}")


def exposure_display_name(exposed: bool) -> str:
    return EXPOSURE_LABELS.get(exposed, str(exposed))


def evaluator_display_name(evaluator_id: str) -> str:
    return EVALUATOR_LABELS.get(str(evaluator_id), str(evaluator_id))


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


def annotate_injected_error_round(ax, error_idx, y_top=5.0, text_y=5.25):
    ax.axvline(
        error_idx,
        linestyle="--",
        linewidth=1,
        zorder=3,
    )

    ax.annotate(
        "Injected AI error",
        xy=(error_idx, y_top),
        xytext=(error_idx + 0.12, text_y),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=8,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.8,
        },
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.9,
        },
    )


def is_ai_supported_row(row):
    if "isAiSupportedWorkflow" in row.index:
        return parse_bool(row["isAiSupportedWorkflow"])

    return row.get("workflow") in ["ai", "human_ai", "ai_human"]


def normalize_ranking(items):
    """Normalise ranking entries to configured workflow identifiers."""
    result = []

    for item in items:
        normalized = str(item).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace("→", "_")
        normalized = re.sub(r"_+", "_", normalized)

        if normalized in WORKFLOW_ORDER:
            result.append(normalized)

    return result


def parse_workflow_ranking(value):
    """Parse JSON and legacy string representations of a workflow ranking."""
    if pd.isna(value):
        return []

    raw = str(value).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ranking = []

            for item in parsed:
                if isinstance(item, str):
                    ranking.append(item)
                elif isinstance(item, dict) and "workflow" in item:
                    ranking.append(str(item["workflow"]))

            return normalize_ranking(ranking)
    except json.JSONDecodeError:
        pass

    raw = raw.strip("{}[]()")
    return normalize_ranking(re.split(r"[>,;|\n]+", raw))


def build_valid_ranking_rows(feedback_df):
    """Return complete rankings plus an audit table for invalid entries."""
    required = {"sessionId", "workflowRanking"}

    if feedback_df.empty or not require_columns(
        feedback_df,
        required,
        "workflow preference ranking",
    ):
        return pd.DataFrame(), pd.DataFrame()

    feedback = feedback_df.drop_duplicates(
        "sessionId",
        keep="last",
    ).copy()
    ranking_rows = []
    audit_rows = []
    expected_workflows = set(WORKFLOW_ORDER)

    for _, row in feedback.iterrows():
        ranking = parse_workflow_ranking(row["workflowRanking"])
        is_valid = (
            len(ranking) == len(WORKFLOW_ORDER) and set(ranking) == expected_workflows
        )

        audit_rows.append(
            {
                "sessionId": row["sessionId"],
                "validRanking": is_valid,
                "parsedWorkflowCount": len(ranking),
                "ranking": " > ".join(ranking),
            }
        )

        if not is_valid:
            continue

        for rank, workflow in enumerate(ranking, start=1):
            ranking_rows.append(
                {
                    "sessionId": row["sessionId"],
                    "workflow": workflow,
                    "rank": rank,
                }
            )

    return pd.DataFrame(ranking_rows), pd.DataFrame(audit_rows)


def ranking_summary(ranking_rows):
    """Build rank counts and mean rank for complete valid rankings."""
    rank_counts = (
        ranking_rows.groupby(["workflow", "rank"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=WORKFLOW_ORDER,
            columns=range(1, len(WORKFLOW_ORDER) + 1),
            fill_value=0,
        )
    )
    rank_counts["meanRank"] = (
        ranking_rows.groupby("workflow")["rank"].mean().reindex(WORKFLOW_ORDER)
    )

    return rank_counts


def quality_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
) -> pd.DataFrame:
    """Calculate descriptive mean, spread, and normal-approximation 95% CI."""
    summary = (
        dataframe.groupby(group_columns, dropna=False)[metric]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ciLow"] = summary["mean"] - CI_Z_VALUE * summary["se"]
    summary["ciHigh"] = summary["mean"] + CI_Z_VALUE * summary["se"]
    summary.loc[summary["count"] < 2, ["std", "se", "ciLow", "ciHigh"]] = np.nan

    return summary


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a 95% Wilson interval in percentage points."""
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0, (centre - margin) * 100), min(100, (centre + margin) * 100)


def pass_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarise complete constraint-pass rates with Wilson intervals."""
    data = dataframe.dropna(subset=["passedNumeric"]).copy()
    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(group_columns)["passedNumeric"]
        .agg(totalRounds="count", passedRounds="sum")
        .reset_index()
    )
    summary["passedRounds"] = summary["passedRounds"].astype(int)
    summary["passRatePercent"] = summary["passedRounds"] / summary["totalRounds"] * 100

    intervals = summary.apply(
        lambda row: wilson_interval(
            int(row["passedRounds"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals
    return summary


def parse_requirement_results(value) -> list[dict]:
    """Safely parse the JSON requirement-results field."""
    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def add_passed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary complete-constraint outcome used by exposure figures."""
    prepared = df.copy()

    if "passedNumeric" in prepared.columns:
        prepared["passedNumeric"] = pd.to_numeric(
            prepared["passedNumeric"],
            errors="coerce",
        )
        return prepared

    if "passed" not in prepared.columns:
        prepared["passedNumeric"] = np.nan
        return prepared

    prepared["passedNumeric"] = prepared["passed"].apply(
        lambda value: (
            1.0
            if parse_bool_or_none(value) is True
            else 0.0
            if parse_bool_or_none(value) is False
            else np.nan
        )
    )

    return prepared
