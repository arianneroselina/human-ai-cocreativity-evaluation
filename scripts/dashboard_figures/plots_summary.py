import json
import os
from collections import defaultdict

import pandas as pd
from openai import OpenAI

from scripts.config import TABLE_DIR, WORKFLOW_LABELS


OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_COMMENT_CHAR_LIMIT = 12000
OPENAI_MAX_COMMENTS_PER_WORKFLOW = 200

WORKFLOW_COLUMNS = ["workflow", "workflowName", "workflowLabel"]
ROUND_COMMENT_COLUMNS = ["roundComment", "comment"]
FINAL_FEEDBACK_COMMENT_COLUMNS = ["comments", "rankingReason", "comment"]


def _detect_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def _clean_text(value):
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().split())


def _normalize_workflow(value):
    workflow = _clean_text(value)

    if not workflow:
        return "unknown"

    return workflow


def _workflow_label(workflow):
    return WORKFLOW_LABELS.get(workflow, workflow)


def _collect_round_comments(df):
    comments = []

    if df is None or df.empty:
        return comments

    workflow_column = _detect_column(df, WORKFLOW_COLUMNS)
    available_comment_columns = [
        column for column in ROUND_COMMENT_COLUMNS
        if column in df.columns
    ]

    if not workflow_column or not available_comment_columns:
        return comments

    for _, row in df.iterrows():
        workflow = _normalize_workflow(row[workflow_column])

        for column in available_comment_columns:
            comment = _clean_text(row[column])

            if comment:
                comments.append({
                    "source": "round",
                    "workflow": workflow,
                    "workflowLabel": _workflow_label(workflow),
                    "comment": comment,
                })

    return comments


def _collect_final_feedback_comments(feedback_df):
    comments = []

    if feedback_df is None or feedback_df.empty:
        return comments

    available_comment_columns = [
        column for column in FINAL_FEEDBACK_COMMENT_COLUMNS
        if column in feedback_df.columns
    ]

    if not available_comment_columns:
        return comments

    for _, row in feedback_df.iterrows():
        for column in available_comment_columns:
            comment = _clean_text(row[column])

            if comment:
                comments.append({
                    "source": "final_feedback",
                    "workflow": "final_feedback",
                    "workflowLabel": "Final feedback",
                    "comment": comment,
                })

    return comments


def _group_comments_by_workflow(comment_rows):
    grouped = defaultdict(list)

    for row in comment_rows:
        grouped[row["workflow"]].append(row["comment"])

    return grouped


def _build_comment_payload(comments):
    lines = []
    total_chars = 0

    for comment in comments[:OPENAI_MAX_COMMENTS_PER_WORKFLOW]:
        line = f"- {comment}"

        if lines and total_chars + len(line) + 1 > OPENAI_COMMENT_CHAR_LIMIT:
            break

        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines)


def _summarize_comments(client, workflow_label, comments):
    payload = _build_comment_payload(comments)

    if not payload:
        return "No feedback comments were available."

    response = client.chat.completions.create(
        model=OPENAI_SUMMARY_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize participant feedback for an academic research dashboard. "
                    "Use only the provided comments. "
                    "Write 3 to 5 concise sentences. "
                    "Cover the overall sentiment, main strengths, main frustrations, "
                    "and concrete improvement suggestions. "
                    "Do not use bullet points or markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Feedback group: {workflow_label}\n\n"
                    "Comments:\n"
                    f"{payload}"
                ),
            },
        ],
    )

    summary = response.choices[0].message.content
    return summary.strip() if summary else "Summary unavailable."


def generate_workflow_feedback_summaries(df, feedback_df):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    comment_rows = [
        *_collect_round_comments(df),
        *_collect_final_feedback_comments(feedback_df),
    ]

    if not comment_rows:
        return

    comments_df = pd.DataFrame(comment_rows)
    comments_df.to_csv(TABLE_DIR / "workflow_feedback_comments.csv", index=False)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Skipping feedback summaries: OPENAI_API_KEY is not set.")
        return

    client = OpenAI(api_key=api_key)
    comments_by_workflow = _group_comments_by_workflow(comment_rows)

    summary_rows = []

    for workflow, comments in sorted(comments_by_workflow.items()):
        workflow_label = _workflow_label(workflow)

        summary_rows.append({
            "workflow": workflow,
            "workflowLabel": workflow_label,
            "commentCount": len(comments),
            "summary": _summarize_comments(client, workflow_label, comments),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["workflow", "commentCount"],
        ascending=[True, False],
    )

    csv_path = TABLE_DIR / "workflow_feedback_summaries.csv"
    json_path = TABLE_DIR / "workflow_feedback_summaries.json"

    summary_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2, ensure_ascii=False)

    print(f"Generated feedback summaries: {csv_path}")
