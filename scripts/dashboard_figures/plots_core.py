import json
import re

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_LABELS, TABLE_DIR, WORKFLOW_ORDER
from scripts.dashboard_figures.utils import save_figure, workflow_label


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

            return normalize_ranking(ranking)

    except json.JSONDecodeError:
        pass

    parts = re.split(r"[>,;|\n]+", raw)
    return normalize_ranking(parts)


def normalize_ranking(items):
    result = []

    for item in items:
        normalized = str(item).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace("→", "_")
        normalized = normalized.replace("__", "_")

        if normalized in WORKFLOW_ORDER:
            result.append(normalized)

    return result


def plot_workflow_distribution(df):
    choice_df = df[df["roundIndex"] >= 5].copy()

    if choice_df.empty:
        return

    counts = (
        choice_df
        .groupby(["roundIndex", "workflow"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=WORKFLOW_ORDER, fill_value=0)
    )

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.rename(columns=WORKFLOW_LABELS)

    percentages.to_csv(TABLE_DIR / "workflow_distribution_choice_rounds.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    percentages.plot(kind="bar", ax=ax)

    ax.set_title("Workflow Distribution in Choice Rounds")
    ax.set_xlabel("Choice round")
    ax.set_ylabel("Share of participants (%)")
    ax.legend(title="Workflow")
    ax.set_ylim(0, 100)

    save_figure(
        fig,
        "01_workflow_distribution_choice_rounds",
        "Workflow Distribution in Choice Rounds",
        "Share of selected workflows in rounds 5–7.",
    )


def plot_total_workflow_usage_counts(df):
    """
    Shows the total number of times each workflow was used.

    This gives a simple overview of workflow usage across all recorded rounds.
    """

    if "workflow" not in df.columns:
        return

    summary = (
        df["workflow"]
        .value_counts()
        .reindex(WORKFLOW_ORDER)
        .dropna()
        .astype(int)
    )

    if summary.empty:
        return

    summary.rename(index=WORKFLOW_LABELS).to_csv(
        TABLE_DIR / "total_workflow_usage_counts.csv",
        header=["count"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(summary.rename(index=WORKFLOW_LABELS).index, summary.values)

    ax.set_title("Total Workflow Usage")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Number of rounds")
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.bar_label(bars, padding=3)

    save_figure(
        fig,
        "01_total_workflow_usage_counts",
        "Total Workflow Usage",
        "Total number of rounds in which each workflow was used.",
    )


def plot_workflow_transitions(df):
    if "participantId" not in df.columns:
        return

    transition_rows = []

    for _, participant_df in df.groupby("participantId"):
        participant_df = participant_df.sort_values("roundIndex")

        rows = participant_df[["roundIndex", "workflow"]].dropna().to_dict("records")

        for index in range(len(rows) - 1):
            current = rows[index]
            next_row = rows[index + 1]

            from_round = int(current["roundIndex"])
            to_round = int(next_row["roundIndex"])

            if from_round < 4 or to_round < 5:
                continue

            transition_rows.append({
                "fromRound": from_round,
                "toRound": to_round,
                "fromWorkflow": current["workflow"],
                "toWorkflow": next_row["workflow"],
            })

    if not transition_rows:
        return

    transition_df = pd.DataFrame(transition_rows)

    summary = (
        transition_df
        .groupby(["fromRound", "toRound", "fromWorkflow", "toWorkflow"])
        .size()
        .reset_index(name="count")
        .sort_values(["fromRound", "toRound", "count"], ascending=[True, True, False])
    )

    summary["from"] = (
            "R" + summary["fromRound"].astype(str) + " " +
            summary["fromWorkflow"].map(WORKFLOW_LABELS)
    )

    summary["to"] = (
            "R" + summary["toRound"].astype(str) + " " +
            summary["toWorkflow"].map(WORKFLOW_LABELS)
    )

    table_df = summary[["from", "to", "count"]]
    table_df.to_csv(TABLE_DIR / "workflow_transitions.csv", index=False)

    fig_height = max(3.5, 0.35 * len(table_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=["From", "To", "Count"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    ax.set_title("Workflow Transition Table", pad=16)

    save_figure(
        fig,
        "02_workflow_transition_table",
        "Workflow Transition Table",
        "Workflow changes from round 4→5, 5→6, and 6→7.",
    )


def plot_mean_quality_by_workflow(df):
    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        metric = "meanOverallQuality"

    if metric not in df.columns or df[metric].dropna().empty:
        return

    summary = (
        df
        .dropna(subset=[metric])
        .groupby("workflow")[metric]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna()
    )

    if summary.empty:
        return

    summary.rename(index=WORKFLOW_LABELS).to_csv(
        TABLE_DIR / "mean_quality_by_workflow.csv",
        header=["meanQuality"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    summary.rename(index=WORKFLOW_LABELS).plot(kind="bar", ax=ax)

    ax.set_title("Mean Output Quality by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean quality score")
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        "03_mean_quality_by_workflow",
        "Mean Quality by Workflow",
        "Mean evaluator-rated output quality by workflow.",
    )


def plot_subjective_feedback_by_workflow(df):
    columns = ["satisfactionResult", "frustration"]

    available_columns = [column for column in columns if column in df.columns]

    if not available_columns:
        return

    summary = (
        df
        .groupby("workflow")[available_columns]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .rename(index=WORKFLOW_LABELS)
    )

    if summary.empty:
        return

    summary = summary.rename(columns={
        "satisfactionResult": "Satisfaction",
        "frustration": "Frustration",
    })

    summary.to_csv(TABLE_DIR / "subjective_feedback_by_workflow.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    summary.plot(kind="bar", ax=ax)

    ax.set_title("Mean Satisfaction and Frustration by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean rating")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Measure")

    save_figure(
        fig,
        "04_satisfaction_frustration_by_workflow",
        "Mean Satisfaction and Frustration by Workflow",
        "Participant-reported satisfaction and frustration after each round.",
    )


def plot_ai_performance_over_rounds(df):
    if "aiPerformanceOverall" not in df.columns:
        return

    ai_df = df[
        (df["workflow"] != "human") &
        (df["aiPerformanceOverall"].notna())
        ].copy()

    if ai_df.empty:
        return

    summary = (
        ai_df
        .groupby(["condition", "roundIndex"])["aiPerformanceOverall"]
        .mean()
        .reset_index()
        .sort_values(["condition", "roundIndex"])
    )

    summary.to_csv(TABLE_DIR / "ai_performance_over_rounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for condition, condition_df in summary.groupby("condition"):
        ax.plot(
            condition_df["roundIndex"],
            condition_df["aiPerformanceOverall"],
            marker="o",
            label=str(condition),
        )

    ax.set_title("AI Performance Rating over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean AI performance rating")
    ax.legend(title="Condition")

    save_figure(
        fig,
        "05_ai_performance_over_rounds",
        "Trust / AI Performance over Rounds",
        "Mean AI performance rating over rounds, split by condition.",
    )


def plot_constraint_rate_by_workflow(df):
    if "passed" not in df.columns:
        return

    constraint_df = df.copy()

    constraint_df["passedNumeric"] = constraint_df["passed"].map({
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "t": 1,
        "f": 0,
        1: 1,
        0: 0,
    })

    summary = (
            constraint_df
            .groupby("workflow")["passedNumeric"]
            .mean()
            .reindex(WORKFLOW_ORDER)
            .dropna() * 100
    )

    if summary.empty:
        return

    summary.rename(index=WORKFLOW_LABELS).to_csv(
        TABLE_DIR / "passed_constraint_rate_by_workflow.csv",
        header=["passedRatePercent"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    summary.rename(index=WORKFLOW_LABELS).plot(kind="bar", ax=ax)

    ax.set_title("Passed Constraint Rate by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Passed rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        "06_passed_constraint_rate_by_workflow",
        "Passed Constraint Rate by Workflow",
        "Percentage of rounds that passed the task constraints.",
    )


def plot_quality_vs_time(df):
    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        return

    time_column = "effectiveTimeMinutes"

    if time_column not in df.columns:
        df[time_column] = df["timeMs"] / 60000

    plot_df = df.dropna(subset=[time_column, metric, "workflow"]).copy()

    if plot_df.empty:
        return

    plot_df[[
        "roundId",
        "participantId",
        "roundIndex",
        "workflow",
        time_column,
        metric,
    ]].to_csv(TABLE_DIR / "quality_vs_time_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for workflow in WORKFLOW_ORDER:
        workflow_df = plot_df[plot_df["workflow"] == workflow]

        if workflow_df.empty:
            continue

        ax.scatter(
            workflow_df[time_column],
            workflow_df[metric],
            label=workflow_label(workflow),
            alpha=0.75,
        )

    ax.set_title("Output Quality vs. Completion Time")
    ax.set_xlabel("Time used (minutes)")
    ax.set_ylabel("Quality composite")
    ax.legend(title="Workflow")

    save_figure(
        fig,
        "07_quality_vs_time_scatterplot",
        "Quality vs Time Scatterplot",
        "Relationship between completion time and externally rated output quality.",
    )


def plot_final_workflow_ranking(feedback_df):
    if feedback_df.empty or "workflowRanking" not in feedback_df.columns:
        return

    ranking_rows = []

    for _, row in feedback_df.iterrows():
        ranking = parse_workflow_ranking(row["workflowRanking"])

        for rank_index, workflow in enumerate(ranking, start=1):
            ranking_rows.append({
                "workflow": workflow,
                "rank": rank_index,
                "isFirstChoice": rank_index == 1,
            })

    if not ranking_rows:
        return

    ranking_df = pd.DataFrame(ranking_rows)

    first_choice = (
        ranking_df
        .groupby("workflow")["isFirstChoice"]
        .sum()
        .reindex(WORKFLOW_ORDER)
        .fillna(0)
    )

    average_rank = (
        ranking_df
        .groupby("workflow")["rank"]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna()
    )

    final_table = pd.DataFrame({
        "firstChoiceCount": first_choice,
        "averageRank": average_rank,
    }).rename(index=WORKFLOW_LABELS)

    final_table.to_csv(TABLE_DIR / "final_workflow_ranking.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    first_choice.rename(index=WORKFLOW_LABELS).plot(kind="bar", ax=ax)

    ax.set_title("Final Workflow Preference")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Number of first-choice rankings")
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        "08_final_workflow_first_choice",
        "Final Workflow Ranking",
        "Number of participants who ranked each workflow first.",
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    average_rank.rename(index=WORKFLOW_LABELS).plot(kind="bar", ax=ax)

    ax.set_title("Average Final Workflow Rank")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Average rank, lower is better")
    ax.tick_params(axis="x", rotation=0)
    ax.invert_yaxis()

    save_figure(
        fig,
        "08b_final_workflow_average_rank",
        "Average Final Workflow Rank",
        "Average rank of each workflow in the final feedback.",
    )
