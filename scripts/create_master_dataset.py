"""Build the round-level master dataset from exported participant folders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.config import (
    AI_SUPPORTED_WORKFLOWS,
    INJECTED_ERROR_ROUND_INDEX,
    INPUTS_DIR,
    MAIN_ROUND_INDICES,
    MASTER_DATASET_PATH,
    POEM_SCORES_PATH,
    PRACTICE_ROUND_INDICES,
    TLX_METRICS,
)
from scripts.utils import parse_bool_or_none


CONSTRAINT_SUMMARY_COLUMNS = [
    "constraintCount",
    "constraintPassedCount",
    "constraintScore",
]


def empty_constraint_stats() -> pd.Series:
    """Return an empty constraint-summary row with the canonical schema."""
    return pd.Series({column: None for column in CONSTRAINT_SUMMARY_COLUMNS})


def extract_constraint_stats(value: object) -> pd.Series:
    """Summarise the JSON-encoded requirement results for one round."""
    if pd.isna(value) or not str(value).strip():
        return empty_constraint_stats()

    try:
        requirements = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return empty_constraint_stats()

    if not isinstance(requirements, list):
        return empty_constraint_stats()

    total = len(requirements)
    passed = sum(
        1
        for item in requirements
        if isinstance(item, dict) and parse_bool_or_none(item.get("passed")) is True
    )

    return pd.Series(
        {
            "constraintCount": total,
            "constraintPassedCount": passed,
            "constraintScore": passed / total * 100 if total else None,
        }
    )


def read_first_row(csv_path: Path) -> dict[str, object] | None:
    """Read the first row of a CSV file, returning ``None`` when unavailable."""
    if not csv_path.exists():
        return None

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        return None

    return dataframe.iloc[0].to_dict()


def load_participant_folder(folder: Path) -> pd.DataFrame | None:
    """Load and merge one participant folder's session, round, and feedback data."""
    session_path = folder / "Session.csv"
    round_path = folder / "Round.csv"
    feedback_path = folder / "RoundFeedback.csv"

    if not session_path.exists() or not round_path.exists():
        print(f"Skipping {folder.name}: missing Session.csv or Round.csv")
        return None

    session_row = read_first_row(session_path)
    if session_row is None:
        print(f"Skipping {folder.name}: Session.csv is empty")
        return None

    rounds = pd.read_csv(round_path)
    if rounds.empty:
        print(f"Skipping {folder.name}: Round.csv is empty")
        return None

    rounds = rounds.rename(columns={"id": "roundId", "index": "roundIndex"})
    participant_id = session_row.get("participantId")
    if pd.isna(participant_id):
        participant_id = folder.name

    rounds["participantId"] = participant_id
    rounds["studySessionId"] = session_row.get("id")

    if "passed" in rounds.columns:
        rounds["passed"] = rounds["passed"].apply(parse_bool_or_none).astype("boolean")

    if "requirementResults" in rounds.columns:
        constraint_stats = rounds["requirementResults"].apply(extract_constraint_stats)
        rounds = pd.concat([rounds, constraint_stats], axis=1)
    else:
        for column in CONSTRAINT_SUMMARY_COLUMNS:
            rounds[column] = None

    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path).rename(
            columns={"id": "roundFeedbackId", "comment": "roundComment"}
        )
        merge_columns = {"sessionId", "roundIndex"}
        if merge_columns.issubset(rounds.columns) and merge_columns.issubset(
            feedback.columns
        ):
            feedback = feedback.drop_duplicates(
                subset=["sessionId", "roundIndex"],
                keep="last",
            )
            rounds = rounds.merge(
                feedback,
                on=["sessionId", "roundIndex"],
                how="left",
                suffixes=("", "_feedback"),
                validate="many_to_one",
            )
        else:
            print(
                f"Warning: {feedback_path} could not be merged because "
                "sessionId or roundIndex is missing."
            )

    return rounds


def add_error_exposure_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add participant-level injected-error exposure based on Main Round 1.

    Participants without an observed Main Round 1 receive a missing exposure
    value rather than being incorrectly classified as not exposed.
    """
    prepared = dataframe.copy()
    prepared["roundIndex"] = pd.to_numeric(prepared["roundIndex"], errors="coerce")
    prepared["isAiSupportedWorkflow"] = prepared["workflow"].isin(
        AI_SUPPORTED_WORKFLOWS
    )

    exposure_by_participant = (
        prepared.loc[prepared["roundIndex"].eq(INJECTED_ERROR_ROUND_INDEX)]
        .groupby("participantId", dropna=False)["isAiSupportedWorkflow"]
        .any()
    )
    prepared["errorExposed"] = prepared["participantId"].map(exposure_by_participant)
    prepared["errorExposed"] = prepared["errorExposed"].astype("boolean")

    return prepared


def assign_study_phase(round_index: object) -> str | None:
    """Map a round index to the canonical study phase."""
    numeric_round = pd.to_numeric(round_index, errors="coerce")
    if pd.isna(numeric_round):
        return None

    integer_round = int(numeric_round)
    if integer_round in PRACTICE_ROUND_INDICES:
        return "practice"
    if integer_round in MAIN_ROUND_INDICES:
        return "main"
    return None


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two numeric series while treating non-positive time as missing."""
    numeric_numerator = pd.to_numeric(numerator, errors="coerce")
    numeric_denominator = pd.to_numeric(denominator, errors="coerce")
    valid_denominator = numeric_denominator.where(numeric_denominator.gt(0))
    return numeric_numerator / valid_denominator


def build_master_dataset(inputs_dir: Path = INPUTS_DIR) -> pd.DataFrame:
    """Load all participant folders and return the complete master dataframe."""
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {inputs_dir}")

    participant_frames = []
    for folder in sorted(inputs_dir.iterdir()):
        if not folder.is_dir():
            continue

        participant_rounds = load_participant_folder(folder)
        if participant_rounds is not None:
            participant_frames.append(participant_rounds)

    if not participant_frames:
        raise RuntimeError("No participant data found.")

    master = pd.concat(participant_frames, ignore_index=True)

    numeric_columns = ["roundIndex", "timeMs", "wordCount", "charCount"]
    for column in numeric_columns:
        if column not in master.columns:
            master[column] = np.nan
        master[column] = pd.to_numeric(master[column], errors="coerce")

    master["phase"] = master["roundIndex"].apply(assign_study_phase)

    tlx_columns = list(TLX_METRICS)
    for column in tlx_columns:
        if column not in master.columns:
            master[column] = np.nan
        master[column] = pd.to_numeric(master[column], errors="coerce")
    master["rawNasaTlxScore"] = master[tlx_columns].mean(axis=1)

    master = add_error_exposure_columns(master)

    master["effectiveTimeMinutes"] = (
        pd.to_numeric(master.get("timeMs"), errors="coerce") / 60000
    )
    master["wordsPerMinute"] = safe_rate(
        master.get("wordCount"), master["effectiveTimeMinutes"]
    )
    master["charsPerMinute"] = safe_rate(
        master.get("charCount"), master["effectiveTimeMinutes"]
    )

    if POEM_SCORES_PATH.exists():
        poem_scores = pd.read_csv(POEM_SCORES_PATH)
        if "roundId" in master.columns and "poemId" in poem_scores.columns:
            master = master.merge(
                poem_scores,
                left_on="roundId",
                right_on="poemId",
                how="left",
                suffixes=("", "_rating"),
                validate="many_to_one",
            )
        else:
            print(
                "Warning: poem scores were not merged because roundId or poemId "
                "is missing."
            )
    else:
        print(f"Warning: {POEM_SCORES_PATH} not found. Ratings were not merged.")

    if "meanOverallQuality" in master.columns:
        master["qualityPerMinute"] = safe_rate(
            master["meanOverallQuality"], master["effectiveTimeMinutes"]
        )

    return master.sort_values(["participantId", "roundIndex"])


def main() -> None:
    """Build and export the master dataset."""
    master = build_master_dataset()
    MASTER_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_DATASET_PATH, index=False)

    print(f"Created {MASTER_DATASET_PATH}")
    print(f"Rows: {len(master)}")
    print(f"Participants: {master['participantId'].nunique()}")
    print(f"Expected rows for 24 participants: {24 * 7}")


if __name__ == "__main__":
    main()
