import re

import pandas as pd

from scripts.config import MASTER_DATASET_PATH, WORKFLOW_LABELS, TABLE_DIR, PARTICIPANT_COLUMN_ALIASES, \
    PARTICIPANT_LIKERT_COLUMNS, INPUTS_DIR
from scripts.dashboard_figures.utils import ensure_numeric


def load_master_dataset():
    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_DATASET_PATH} not found. Run make process-data first."
        )

    df = pd.read_csv(MASTER_DATASET_PATH)

    ensure_numeric(df, [
        "roundIndex",
        "participantId",
        "timeMs",
        "wordCount",
        "charCount",
        "effectiveTimeMinutes",
        "qualityComposite",
        "meanOverallQuality",
        "satisfactionResult",
        "frustration",
        "effort",
        "performance",
        "aiPerformanceOverall",
        "aiUnderstanding",
        "aiCollaboration",
        "aiCreativitySupport",
        "constraintScore",
    ])

    if "workflow" in df.columns:
        df["workflowLabel"] = df["workflow"].map(WORKFLOW_LABELS).fillna(df["workflow"])

    if "condition" not in df.columns:
        df["condition"] = "All participants"
    else:
        df["condition"] = df["condition"].fillna("All participants")

    return df


def load_final_feedback():
    rows = []

    if not INPUTS_DIR.exists():
        return pd.DataFrame()

    for folder in INPUTS_DIR.iterdir():
        feedback_path = folder / "Feedback.csv"

        if folder.is_dir() and feedback_path.exists():
            feedback = pd.read_csv(feedback_path)
            feedback["sourceFolder"] = folder.name
            rows.append(feedback)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_participant_info():
    rows = []

    if not INPUTS_DIR.exists():
        return pd.DataFrame()

    for folder in INPUTS_DIR.iterdir():
        if not folder.is_dir():
            continue

        for csv_path in folder.glob("*.csv"):
            if csv_path.name.lower() == "feedback.csv":
                continue

            try:
                raw_df = pd.read_csv(csv_path)
            except Exception:
                continue

            clean_df = pd.DataFrame()

            for target_column, aliases in PARTICIPANT_COLUMN_ALIASES.items():
                source_column = find_source_column(raw_df.columns, aliases)

                if source_column is not None:
                    clean_df[target_column] = raw_df[source_column]

            if "age" not in clean_df.columns and "participantId" not in clean_df.columns:
                continue

            clean_df["sourceFolder"] = folder.name
            clean_df["sourceFile"] = csv_path.name

            rows.append(clean_df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    if "participantId" not in df.columns:
        df["participantId"] = df["sourceFolder"]

    for column in ["gender", "education", "nativeLanguage", "englishLevel"]:
        if column in df.columns:
            df[column] = df[column].apply(clean_category)

    if "genderOther" in df.columns:
        df["gender"] = df.apply(
            lambda row: replace_other_value(row, "gender", "genderOther"),
            axis=1,
        )

    if "educationOther" in df.columns:
        df["education"] = df.apply(
            lambda row: replace_other_value(row, "education", "educationOther"),
            axis=1,
        )

    if "englishLevelOther" in df.columns:
        df["englishLevel"] = df.apply(
            lambda row: replace_other_value(row, "englishLevel", "englishLevelOther"),
            axis=1,
        )

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    for column in PARTICIPANT_LIKERT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].apply(parse_likert)

    df.to_csv(TABLE_DIR / "participant_info_clean.csv", index=False)

    return df


def normalize_column_name(column):
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def find_source_column(columns, aliases):
    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    return None


def replace_other_value(row, main_column, other_column):
    value = clean_category(row.get(main_column))
    other_value = clean_category(row.get(other_column))

    if value and value.lower() == "other" and other_value:
        return other_value

    return value


def clean_category(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    return text if text else None


def parse_likert(value):
    if pd.isna(value):
        return None

    match = re.search(r"\d+", str(value))

    if not match:
        return None

    return int(match.group(0))

