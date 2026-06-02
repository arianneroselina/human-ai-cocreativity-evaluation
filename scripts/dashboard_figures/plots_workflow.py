import json
import re

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_LABELS, TABLE_DIR, WORKFLOW_ORDER
from scripts.dashboard_figures.utils import save_figure


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

    raw = raw.strip("{}[]()")

    parts = re.split(r"[>,;|\n,]+", raw)

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


def plot_total_workflow_usage_counts(df):
    """
    Shows the total number of times each workflow was used.

    This gives a simple overview of workflow usage across all recorded rounds.
    """
    slug = "01_total_workflow_usage_counts"

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
        TABLE_DIR / f"{slug}.csv",
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
        slug,
        "Total Workflow Usage",
        "Total number of rounds in which each workflow was used.",
    )


def plot_workflow_distribution(df):
    slug = "02_workflow_distribution_main_rounds"

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

    percentages.to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    percentages.plot(kind="bar", ax=ax)

    ax.set_title("Workflow Distribution in Main Rounds")
    ax.set_xlabel("Main round")
    ax.set_ylabel("Share of participants (%)")
    ax.legend(title="Workflow")
    ax.set_ylim(0, 100)

    save_figure(
        fig,
        slug,
        "Workflow Distribution in Main Rounds",
        "Share of selected workflows in rounds 5–7.",
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

    for (from_round, to_round), step_df in transition_df.groupby(["fromRound", "toRound"]):
        slug = f"03_workflow_transition_r{from_round}_to_r{to_round}"

        matrix = (
            step_df
            .groupby(["fromWorkflow", "toWorkflow"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=WORKFLOW_ORDER, columns=WORKFLOW_ORDER, fill_value=0)
        )

        matrix = matrix.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS)

        matrix.to_csv(TABLE_DIR / f"{slug}.csv")

        fig, ax = plt.subplots(figsize=(6.8, 5.4))

        image = ax.imshow(matrix.values)

        ax.set_title(f"Workflow Transitions R{from_round} → R{to_round}")
        ax.set_xlabel(f"Workflow in round {to_round}")
        ax.set_ylabel(f"Workflow in round {from_round}")

        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=30, ha="right")

        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)

        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = int(matrix.iloc[row_index, col_index])
                ax.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, label="Number of participants")

        save_figure(
            fig,
            slug,
            f"Workflow Transitions R{from_round} to R{to_round}",
            f"Workflow transition counts from round {from_round} to round {to_round}.",
        )


def plot_final_workflow_ranking(feedback_df):
    slug = "04_final_workflow_ranking_overview"

    if feedback_df.empty or "workflowRanking" not in feedback_df.columns:
        return

    ranking_rows = []

    for _, row in feedback_df.iterrows():
        ranking = parse_workflow_ranking(row["workflowRanking"])

        for rank_index, workflow in enumerate(ranking, start=1):
            ranking_rows.append({
                "workflow": workflow,
                "rank": rank_index,
            })

    if not ranking_rows:
        return

    ranking_df = pd.DataFrame(ranking_rows)

    rank_counts = (
        ranking_df
        .groupby(["workflow", "rank"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=WORKFLOW_ORDER, fill_value=0)
    )

    rank_counts = rank_counts.rename(index=WORKFLOW_LABELS)
    rank_counts.columns = [f"Rank {int(column)}" for column in rank_counts.columns]

    average_rank = (
        ranking_df
        .groupby("workflow")["rank"]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .rename(index=WORKFLOW_LABELS)
    )

    rank_counts["Average rank"] = average_rank.round(2)

    rank_counts.to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=rank_counts.values,
        rowLabels=rank_counts.index,
        colLabels=rank_counts.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    ax.set_title("Final Workflow Ranking Overview", pad=16)

    save_figure(
        fig,
        slug,
        "Final Workflow Ranking Overview",
        "Distribution of final workflow rankings and average rank per workflow.",
    )


def plot_workflow(df, feedback_df):
    plot_total_workflow_usage_counts(df)
    plot_workflow_distribution(df)
    plot_workflow_transitions(df)
    plot_final_workflow_ranking(feedback_df)
