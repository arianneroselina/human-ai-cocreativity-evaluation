"""Workflow-ranking parsing, validation, and summaries."""

from __future__ import annotations

import json
import re

import pandas as pd

from scripts.config import WORKFLOW_ORDER
from scripts.utils import require_columns


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
