import json
import re

import pandas as pd

from scripts.config import (
    INPUTS_DIR,
    MASTER_DATASET_PATH,
    DASHBOARD_FINAL_RANKING_PATH,
    DASHBOARD_COMMENT_THEMES_PATH,
    TABLE_DIR,
    WORKFLOW_ORDER,
)

THEME_KEYWORDS = {
    "AI error / misunderstanding": [
        "error",
        "mistake",
        "wrong",
        "incorrect",
        "misunderstood",
        "not understand",
        "didn't understand",
    ],
    "Control / ownership": ["control", "ownership", "own", "my text", "edit"],
    "Speed / time": ["time", "fast", "quick", "slow", "deadline"],
    "Creativity": ["creative", "creativity", "idea", "inspiration"],
    "Quality": ["quality", "better", "good", "bad", "improve"],
    "Rules / constraints": ["rule", "constraint", "requirement", "forbidden", "required"],
    "Frustration": ["frustrated", "frustrating", "annoying", "stress", "difficult"],
    "Trust": ["trust", "reliable", "confidence", "depend"],
    "Helpfulness": ["helpful", "support", "assist", "useful"],
}


def load_final_feedback():
    rows = []

    if not INPUTS_DIR.exists():
        print(f"Warning: {INPUTS_DIR} not found. Final feedback tables will be empty.")
        return pd.DataFrame()

    for folder in sorted(INPUTS_DIR.iterdir()):
        feedback_path = folder / "Feedback.csv"

        if folder.is_dir() and feedback_path.exists():
            feedback = pd.read_csv(feedback_path)
            feedback["sourceFolder"] = folder.name
            rows.append(feedback)

    if not rows:
        print("Warning: No Feedback.csv files found.")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_master_dataset():
    if not MASTER_DATASET_PATH.exists():
        print(
            f"Warning: {MASTER_DATASET_PATH} not found. "
            "Round comments will not be included in comment theme counts."
        )
        return pd.DataFrame()

    return pd.read_csv(MASTER_DATASET_PATH)


def normalize_ranking_items(items):
    normalized_items = []

    for item in items:
        normalized = str(item).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace("→", "_")
        normalized = normalized.replace("__", "_")

        if normalized in WORKFLOW_ORDER:
            normalized_items.append(normalized)

    return normalized_items


def parse_workflow_ranking(value):
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

            return normalize_ranking_items(ranking)

    except json.JSONDecodeError:
        pass

    raw = raw.strip("{}[]()")
    parts = re.split(r"[>,;|\n,]+", raw)

    return normalize_ranking_items(parts)


def create_final_ranking_table(feedback_df):
    rank_sums = {workflow: 0 for workflow in WORKFLOW_ORDER}
    rank_counts = {workflow: 0 for workflow in WORKFLOW_ORDER}
    first_choice_counts = {workflow: 0 for workflow in WORKFLOW_ORDER}

    if not feedback_df.empty and "workflowRanking" in feedback_df.columns:
        for _, row in feedback_df.iterrows():
            ranking = parse_workflow_ranking(row["workflowRanking"])

            for index, workflow in enumerate(ranking, start=1):
                rank_sums[workflow] += index
                rank_counts[workflow] += 1

                if index == 1:
                    first_choice_counts[workflow] += 1

    rows = []

    for workflow in WORKFLOW_ORDER:
        rank_count = rank_counts[workflow]

        rows.append({
            "workflow": workflow,
            "firstChoiceCount": first_choice_counts[workflow],
            "averageRank": rank_sums[workflow] / rank_count if rank_count > 0 else None,
        })

    return pd.DataFrame(rows)


def clean_text(value):
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().split())


def collect_comments(master_df, feedback_df):
    comments = []

    if not master_df.empty and "roundComment" in master_df.columns:
        comments.extend(master_df["roundComment"].apply(clean_text).tolist())

    if not feedback_df.empty:
        for column in ["comments", "rankingReason"]:
            if column in feedback_df.columns:
                comments.extend(feedback_df[column].apply(clean_text).tolist())

    return [comment for comment in comments if comment]


def create_comment_theme_table(master_df, feedback_df):
    comments = collect_comments(master_df, feedback_df)
    rows = []

    for theme, keywords in THEME_KEYWORDS.items():
        count = 0

        for comment in comments:
            lower_comment = comment.lower()

            if any(keyword in lower_comment for keyword in keywords):
                count += 1

        if count > 0:
            rows.append({
                "theme": theme,
                "count": count,
            })

    return pd.DataFrame(rows).sort_values("count", ascending=False)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    master_df = load_master_dataset()
    feedback_df = load_final_feedback()

    final_ranking_df = create_final_ranking_table(feedback_df)
    comment_themes_df = create_comment_theme_table(master_df, feedback_df)

    final_ranking_df.to_csv(DASHBOARD_FINAL_RANKING_PATH, index=False)
    comment_themes_df.to_csv(DASHBOARD_COMMENT_THEMES_PATH, index=False)

    print(f"Created {DASHBOARD_FINAL_RANKING_PATH}")
    print(f"Rows: {len(final_ranking_df)}")
    print(f"Created {DASHBOARD_COMMENT_THEMES_PATH}")
    print(f"Rows: {len(comment_themes_df)}")


if __name__ == "__main__":
    main()
