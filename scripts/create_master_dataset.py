import json
import pandas as pd

from scripts.config import (
    MASTER_DATASET_PATH,
    POEM_SCORES_PATH,
    INPUTS_DIR,
    ERROR_ROUND_INDEX,
    AI_SUPPORTED_WORKFLOWS,
)
from scripts.utils import parse_bool, parse_bool_or_none


def extract_constraint_stats(value):
    if pd.isna(value) or not str(value).strip():
        return pd.Series(
            {
                "constraintCount": None,
                "constraintPassedCount": None,
                "constraintScore": None,
            }
        )

    try:
        requirements = json.loads(value)
    except json.JSONDecodeError:
        return pd.Series(
            {
                "constraintCount": None,
                "constraintPassedCount": None,
                "constraintScore": None,
            }
        )

    if not isinstance(requirements, list):
        return pd.Series(
            {
                "constraintCount": None,
                "constraintPassedCount": None,
                "constraintScore": None,
            }
        )

    total = len(requirements)
    passed = sum(
        1 for item in requirements if parse_bool_or_none(item.get("passed")) is True
    )

    return pd.Series(
        {
            "constraintCount": total,
            "constraintPassedCount": passed,
            "constraintScore": (passed / total) * 100 if total > 0 else None,
        }
    )


def read_first_row(csv_path):
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    if df.empty:
        return None

    return df.iloc[0].to_dict()


def load_participant_folder(folder):
    session_path = folder / "Session.csv"
    round_path = folder / "Round.csv"
    feedback_path = folder / "RoundFeedback.csv"

    if not session_path.exists() or not round_path.exists():
        print(f"Skipping {folder.name}: missing Session.csv or Round.csv")
        return None

    session_row = read_first_row(session_path)
    rounds = pd.read_csv(round_path)

    if rounds.empty:
        print(f"Skipping {folder.name}: Round.csv is empty")
        return None

    rounds = rounds.rename(
        columns={
            "id": "roundId",
            "index": "roundIndex",
        }
    )

    rounds["participantId"] = session_row.get("participantId")
    rounds["studySessionId"] = session_row.get("id")

    rounds["passed"] = rounds["passed"].apply(parse_bool)

    if "requirementResults" in rounds.columns:
        constraint_stats = rounds["requirementResults"].apply(extract_constraint_stats)
        rounds = pd.concat([rounds, constraint_stats], axis=1)
    else:
        rounds["constraintCount"] = None
        rounds["constraintPassedCount"] = None
        rounds["constraintScore"] = None

    if feedback_path.exists():
        feedback = pd.read_csv(feedback_path)

        feedback = feedback.rename(
            columns={
                "id": "roundFeedbackId",
                "comment": "roundComment",
            }
        )

        rounds = rounds.merge(
            feedback,
            on=["sessionId", "roundIndex"],
            how="left",
            suffixes=("", "_feedback"),
        )

    return rounds


def add_error_exposure_columns(master):
    """
    The AI error happens in the first main round, which is round 5.

    A participant is exposed to the AI error only if the workflow in round 5
    includes AI support.

    AI-supported workflows:
    - ai
    - human_ai
    - ai_human

    Human-only workflow:
    - human
    """

    master["roundIndex"] = pd.to_numeric(master["roundIndex"], errors="coerce")
    master["participantId"] = pd.to_numeric(master["participantId"], errors="coerce")

    master["isAiSupportedWorkflow"] = master["workflow"].isin(AI_SUPPORTED_WORKFLOWS)

    # Keep old column name for compatibility with existing scripts.
    master["isAiWorkflow"] = master["isAiSupportedWorkflow"]

    master["isMixedWorkflow"] = master["workflow"].isin(["human_ai", "ai_human"])

    error_round = master[master["roundIndex"] == ERROR_ROUND_INDEX][
        ["participantId", "isAiSupportedWorkflow"]
    ].copy()

    error_round = error_round.rename(
        columns={
            "isAiSupportedWorkflow": "errorExposed",
        }
    )

    master = master.merge(
        error_round,
        on="participantId",
        how="left",
    )

    master["errorExposed"] = master["errorExposed"].fillna(False).astype(bool)

    master["isErrorRound"] = (master["roundIndex"] == ERROR_ROUND_INDEX) & (
        master["isAiSupportedWorkflow"]
    )

    master["isAfterError"] = (master["errorExposed"]) & (
        master["roundIndex"] > ERROR_ROUND_INDEX
    )

    master["errorRoundIndex"] = master["errorExposed"].apply(
        lambda exposed: ERROR_ROUND_INDEX if exposed else None
    )

    master["errorExposureGroup"] = master["errorExposed"].map(
        {
            True: "error_exposed",
            False: "not_exposed",
        }
    )

    # Compatibility column for older dashboard/figure code.
    master["condition"] = master["errorExposureGroup"]

    return master


all_rounds = []

for folder in sorted(INPUTS_DIR.iterdir()):
    if folder.is_dir():
        participant_rounds = load_participant_folder(folder)

        if participant_rounds is not None:
            all_rounds.append(participant_rounds)

if not all_rounds:
    raise RuntimeError("No participant data found.")

master = pd.concat(all_rounds, ignore_index=True)

master["roundIndex"] = pd.to_numeric(master["roundIndex"], errors="coerce")
master["timeMs"] = pd.to_numeric(master["timeMs"], errors="coerce")
master["wordCount"] = pd.to_numeric(master["wordCount"], errors="coerce")
master["charCount"] = pd.to_numeric(master["charCount"], errors="coerce")

master["phase"] = master["roundIndex"].apply(
    lambda value: "practice" if int(value) <= 4 else "main"
)

master["isPracticeRound"] = master["roundIndex"] <= 4
master["isMainRound"] = master["roundIndex"] >= 5

master["workflowGroup"] = master["workflow"].map(
    {
        "human": "human_only",
        "ai": "ai_only",
        "human_ai": "mixed",
        "ai_human": "mixed",
    }
)

master = add_error_exposure_columns(master)

master["effectiveTimeMinutes"] = master["timeMs"] / 60000
master["wordsPerMinute"] = master["wordCount"] / master["effectiveTimeMinutes"]
master["charsPerMinute"] = master["charCount"] / master["effectiveTimeMinutes"]

if POEM_SCORES_PATH.exists():
    poem_scores = pd.read_csv(POEM_SCORES_PATH)

    master = master.merge(
        poem_scores,
        left_on="roundId",
        right_on="poemId",
        how="left",
        suffixes=("", "_rating"),
    )

    if "qualityComposite" in master.columns:
        master["qualityPerMinute"] = (
            master["qualityComposite"] / master["effectiveTimeMinutes"]
        )
else:
    print(f"Warning: {POEM_SCORES_PATH} not found. Ratings were not merged.")

master = master.sort_values(["participantId", "roundIndex"])

master.to_csv(MASTER_DATASET_PATH, index=False)

print(f"Created {MASTER_DATASET_PATH}")
print(f"Rows: {len(master)}")
print(f"Participants: {master['participantId'].nunique()}")
print(f"Expected rows for 24 participants: {24 * 7}")

print("\nRows per participant:")
print(master.groupby("participantId")["roundId"].count())

print("\nWorkflow counts:")
print(master["workflow"].value_counts())

print("\nError exposure groups:")
print(
    master[["participantId", "errorExposed"]]
    .drop_duplicates()["errorExposed"]
    .value_counts()
)

print("\nError rounds:")
print(master["isErrorRound"].value_counts())
