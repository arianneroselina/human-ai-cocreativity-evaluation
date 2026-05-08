import json
from pathlib import Path
import pandas as pd

INPUTS_DIR = Path("inputs")
PROCESSED_DIR = Path("data/processed")
POEM_SCORES_PATH = PROCESSED_DIR / "poem_scores.csv"
OUTPUT_PATH = PROCESSED_DIR / "master_round_dataset.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_bool(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in ["true", "t", "1", "yes"]:
        return True

    if value in ["false", "f", "0", "no"]:
        return False

    return None


def extract_constraint_stats(value):
    if pd.isna(value) or not str(value).strip():
        return pd.Series({
            "constraintCount": None,
            "constraintPassedCount": None,
            "constraintScore": None,
        })

    try:
        requirements = json.loads(value)
    except json.JSONDecodeError:
        return pd.Series({
            "constraintCount": None,
            "constraintPassedCount": None,
            "constraintScore": None,
        })

    if not isinstance(requirements, list):
        return pd.Series({
            "constraintCount": None,
            "constraintPassedCount": None,
            "constraintScore": None,
        })

    total = len(requirements)
    passed = sum(1 for item in requirements if item.get("passed") is True)

    return pd.Series({
        "constraintCount": total,
        "constraintPassedCount": passed,
        "constraintScore": passed / total if total > 0 else None,
    })


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

    rounds = rounds.rename(columns={
        "id": "roundId",
        "index": "roundIndex",
    })

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

        feedback = feedback.rename(columns={
            "id": "roundFeedbackId",
            "comment": "roundComment",
        })

        merge_keys = ["sessionId", "roundIndex"]

        rounds = rounds.merge(
            feedback,
            on=merge_keys,
            how="left",
            suffixes=("", "_feedback"),
        )

    return rounds


all_rounds = []

for folder in sorted(INPUTS_DIR.iterdir()):
    if folder.is_dir():
        participant_rounds = load_participant_folder(folder)

        if participant_rounds is not None:
            all_rounds.append(participant_rounds)

if not all_rounds:
    raise RuntimeError("No participant data found.")

master = pd.concat(all_rounds, ignore_index=True)

master["phase"] = master["roundIndex"].apply(
    lambda value: "controlled" if int(value) <= 4 else "choice"
)

master["isChoiceRound"] = master["roundIndex"].astype(int) >= 5
master["isAiWorkflow"] = master["workflow"] != "human"
master["isMixedWorkflow"] = master["workflow"].isin(["human_ai", "ai_human"])

master["workflowGroup"] = master["workflow"].map({
    "human": "human_only",
    "ai": "ai_only",
    "human_ai": "mixed",
    "ai_human": "mixed",
})

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

master.to_csv(OUTPUT_PATH, index=False)

print(f"Created {OUTPUT_PATH}")
print(f"Rows: {len(master)}")
print(f"Participants: {master['participantId'].nunique()}")
print(f"Expected rows for 24 participants: {24 * 7}")

print("\nRows per participant:")
print(master.groupby("participantId")["roundId"].count())

print("\nWorkflow counts:")
print(master["workflow"].value_counts())
