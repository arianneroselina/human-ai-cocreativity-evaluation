import textwrap

import os
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
from openai import OpenAI

from scripts.config import TABLE_DIR
from scripts.dashboard_figures.utils import save_figure

OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_COMMENT_CHAR_LIMIT = 12000
OPENAI_MAX_COMMENTS_PER_WORKFLOW = 200

WORKFLOW_COLUMN_CANDIDATES = [
    "workflow",
    "Workflow",
    "workflow_name",
    "workflowName",
    "workflow_id",
    "workflowId",
    "workflow_label",
    "workflowLabel",
    "workflow_type",
    "workflowType",
    "study_workflow",
    "studyWorkflow",
    "assigned_workflow",
    "assignedWorkflow",
    "selected_workflow",
    "selectedWorkflow",
    "creative_workflow",
    "creativeWorkflow",
    "co_creation_workflow",
    "coCreationWorkflow",
    "condition",
    "Condition",
    "condition_name",
    "conditionName",
    "condition_id",
    "conditionId",
    "experimental_condition",
    "experimentalCondition",
    "experiment_condition",
    "experimentCondition",
    "treatment",
    "Treatment",
    "arm",
    "Arm",
    "variant",
    "Variant",
]


def _detect_workflow_column(df):
    for column in WORKFLOW_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    return None


def _normalize_workflow_name(value):
    if pd.isna(value):
        return "Unknown workflow"

    workflow = str(value).strip()
    return workflow if workflow else "Unknown workflow"


def _collect_comments_by_workflow(df, comment_columns):
    comments_by_workflow = defaultdict(list)

    if df is None or df.empty:
        return comments_by_workflow

    workflow_column = _detect_workflow_column(df)
    available_comment_columns = [column for column in comment_columns if column in df.columns]

    if not available_comment_columns:
        return comments_by_workflow

    for _, row in df.iterrows():
        workflow = (
            _normalize_workflow_name(row[workflow_column])
            if workflow_column is not None
            else "All workflows"
        )

        for column in available_comment_columns:
            value = row[column]

            if pd.isna(value):
                continue

            comment = str(value).strip()

            if comment:
                comments_by_workflow[workflow].append(comment)

    return comments_by_workflow


def _merge_comments_by_workflow(*comment_maps):
    merged = defaultdict(list)

    for comment_map in comment_maps:
        for workflow, comments in comment_map.items():
            merged[workflow].extend(comments)

    return merged


def _build_comment_payload(comments):
    payload_lines = []
    total_chars = 0

    for index, comment in enumerate(comments):
        if index >= OPENAI_MAX_COMMENTS_PER_WORKFLOW:
            break

        normalized_comment = " ".join(str(comment).split())

        if not normalized_comment:
            continue

        line = f"- {normalized_comment}"

        if payload_lines and total_chars + len(line) + 1 > OPENAI_COMMENT_CHAR_LIMIT:
            break

        payload_lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(payload_lines)


def _summarize_comments_with_openai(client, workflow, comments):
    comment_payload = _build_comment_payload(comments)

    if not comment_payload:
        return "No feedback comments were available for this workflow."

    response = client.chat.completions.create(
        model=OPENAI_SUMMARY_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You summarize participant feedback for research workflows. "
                    "Use only the provided comments. "
                    "Write 3 to 5 concise sentences that cover: overall sentiment, "
                    "main strengths, main frustrations, and concrete improvement suggestions. "
                    "Do not use bullet points, markdown, or information not present in the comments."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Workflow: {workflow}\n\n"
                    "Summarize the following feedback comments from all participants:\n"
                    f"{comment_payload}"
                ),
            },
        ],
    )

    summary = response.choices[0].message.content
    return summary.strip() if summary else "Summary unavailable."


def plot_workflow_feedback_summaries(df, feedback_df):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set the environment variable before running the dashboard script."
        )

    client = OpenAI(api_key=api_key)

    df_comments_by_workflow = _collect_comments_by_workflow(
        df,
        ["roundComment", "comment"],
    )
    feedback_comments_by_workflow = _collect_comments_by_workflow(
        feedback_df,
        ["comments", "rankingReason", "comment"],
    )

    comments_by_workflow = _merge_comments_by_workflow(
        df_comments_by_workflow,
        feedback_comments_by_workflow,
    )

    comments_by_workflow = {
        workflow: comments
        for workflow, comments in comments_by_workflow.items()
        if comments
    }

    if not comments_by_workflow:
        return

    summary_rows = []

    for workflow in sorted(comments_by_workflow.keys()):
        comments = comments_by_workflow[workflow]
        summary = _summarize_comments_with_openai(client, workflow, comments)

        summary_rows.append(
            {
                "workflow": workflow,
                "comment_count": len(comments),
                "summary": summary,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["comment_count", "workflow"],
        ascending=[False, True],
    )
    summary_df.to_csv(TABLE_DIR / "workflow_feedback_summaries.csv", index=False)

    summary_blocks = []

    for _, row in summary_df.iterrows():
        wrapped_summary = textwrap.fill(
            row["summary"],
            width=110,
            break_long_words=False,
            break_on_hyphens=False,
        )
        summary_blocks.append(
            f"{row['workflow']} ({row['comment_count']} comments)\n{wrapped_summary}"
        )

    figure_text = "\n\n".join(summary_blocks)
    fig_height = max(4.8, 2.4 * len(summary_blocks))

    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title("Workflow Feedback Summaries", loc="left")
    ax.text(
        0.0,
        0.98,
        figure_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )

    save_figure(
        fig,
        "09_workflow_feedback_summaries",
        "Workflow Feedback Summaries",
        "OpenAI-generated summaries of participant feedback comments grouped by workflow.",
    )
