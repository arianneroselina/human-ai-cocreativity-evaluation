import json
import re

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_LABELS, TABLE_DIR, WORKFLOW_ORDER
from scripts.utils import save_figure


def plot_total_workflow_usage_counts(df):
    slug = "01_total_workflow_usage_counts"

    if "workflow" not in df.columns:
        return

    summary = df["workflow"].value_counts().reindex(WORKFLOW_ORDER).dropna().astype(int)

    if summary.empty:
        return

    display_summary = summary.rename(index=WORKFLOW_LABELS)

    export_df = pd.DataFrame(
        {
            "count": display_summary,
            "percentage": (display_summary / display_summary.sum() * 100).round(2),
        }
    )

    export_df.to_csv(TABLE_DIR / f"{slug}.csv")

    def format_pie_label(percent):
        count = int(round(percent * display_summary.sum() / 100))

        if count == 0:
            return ""

        return f"{percent:.1f}%\n(n={count})"

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    wedges, _, _ = ax.pie(
        display_summary.values,
        autopct=format_pie_label,
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1,
        },
        textprops={
            "fontsize": 9,
        },
    )

    ax.legend(
        wedges,
        display_summary.index,
        title="Workflow",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    ax.set_title("Total Workflow Usage")
    ax.axis("equal")

    save_figure(
        fig,
        slug,
        "Total Workflow Usage",
        "Share and number of rounds in which each workflow was used.",
    )


def plot_workflow_distribution(df):
    slug = "02_workflow_distribution_main_rounds"

    required_columns = {"roundIndex", "workflow"}
    if not required_columns.issubset(df.columns):
        return

    choice_df = df[df["roundIndex"] >= 5].copy()
    choice_df = choice_df.dropna(subset=["roundIndex", "workflow"])

    if choice_df.empty:
        return

    if "participantId" in choice_df.columns:
        choice_df = choice_df.drop_duplicates(
            subset=["participantId", "roundIndex"],
            keep="first",
        )

    counts = (
        choice_df.groupby(["roundIndex", "workflow"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=WORKFLOW_ORDER, fill_value=0)
        .sort_index()
    )

    if counts.empty:
        return

    # Sort workflow order according to their frequency in round 5.
    if 5 in counts.index:
        round5_rank = counts.loc[5].sort_values(ascending=False)

        sorted_workflows = [
            workflow for workflow in round5_rank.index if workflow in WORKFLOW_ORDER
        ]
    else:
        sorted_workflows = WORKFLOW_ORDER

    # Keep remaining workflows in the predefined order, if not present in round 5.
    sorted_workflows += [
        workflow for workflow in WORKFLOW_ORDER if workflow not in sorted_workflows
    ]

    counts = counts.reindex(columns=sorted_workflows, fill_value=0)

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    ai_assisted_workflows = [
        workflow for workflow in WORKFLOW_ORDER if workflow != "human"
    ]

    ai_summary = pd.DataFrame(
        {
            "ai_assisted_count": counts[
                [
                    workflow
                    for workflow in ai_assisted_workflows
                    if workflow in counts.columns
                ]
            ].sum(axis=1),
            "total_count": counts.sum(axis=1),
        }
    )

    ai_summary["ai_assisted_percentage"] = (
        ai_summary["ai_assisted_count"] / ai_summary["total_count"] * 100
    ).round(2)

    output_table = percentages.copy()
    output_table.columns = [
        WORKFLOW_LABELS.get(column, column) for column in output_table.columns
    ]

    output_table["AI-assisted count"] = ai_summary["ai_assisted_count"]
    output_table["Total count"] = ai_summary["total_count"]
    output_table["AI-assisted percentage"] = ai_summary["ai_assisted_percentage"]

    output_table.to_csv(TABLE_DIR / f"{slug}.csv")

    plot_percentages = percentages.rename(columns=WORKFLOW_LABELS)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    plot_percentages.plot(
        kind="bar",
        stacked=True,
        ax=ax,
    )

    ax.set_title("Workflow Distribution in Main Rounds")
    ax.set_xlabel("Main round")
    ax.set_ylabel("Share of participants (%)")
    ax.set_ylim(0, 112)
    ax.tick_params(axis="x", rotation=0)

    for index, (round_index, row) in enumerate(ai_summary.iterrows()):
        ax.text(
            index,
            103,
            f"AI-assisted: {row['ai_assisted_percentage']:.1f}%\n"
            f"(n={int(row['ai_assisted_count'])}/{int(row['total_count'])})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Workflow Distribution in Main Rounds",
        "Share of selected workflows in rounds 5–7, with total AI-assisted workflow use annotated above each round. Workflow categories are ordered by their frequency in round 5.",
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

            transition_rows.append(
                {
                    "fromRound": from_round,
                    "toRound": to_round,
                    "fromWorkflow": current["workflow"],
                    "toWorkflow": next_row["workflow"],
                }
            )

    if not transition_rows:
        return

    transition_df = pd.DataFrame(transition_rows)

    for (from_round, to_round), step_df in transition_df.groupby(
        ["fromRound", "toRound"]
    ):
        slug = f"03_workflow_transition_r{from_round}_to_r{to_round}"

        matrix = (
            step_df.groupby(["fromWorkflow", "toWorkflow"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=WORKFLOW_ORDER, columns=WORKFLOW_ORDER, fill_value=0)
        )

        row_totals = matrix.sum(axis=1).replace(0, pd.NA)
        matrix_percentages = matrix.div(row_totals, axis=0) * 100
        matrix_percentages = matrix_percentages.fillna(0)

        matrix = matrix.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS)

        matrix.to_csv(TABLE_DIR / f"{slug}.csv")

        matrix_percentages_renamed = matrix_percentages.rename(
            index=WORKFLOW_LABELS,
            columns=WORKFLOW_LABELS,
        )

        matrix_percentages_renamed.round(2).to_csv(
            TABLE_DIR / f"{slug}_row_percentages.csv"
        )

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
                percentage = matrix_percentages.iloc[row_index, col_index]

                label = f"{value}\n{percentage:.1f}%" if value > 0 else "0"

                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, label="Number of participants")

        save_figure(
            fig,
            slug,
            f"Workflow Transitions R{from_round} to R{to_round}",
            f"Workflow transition counts from round {from_round} to round {to_round}, with row-wise percentages.",
        )


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


def plot_final_workflow_ranking(feedback_df):
    slug = "04_final_workflow_ranking_overview"

    if feedback_df.empty or "workflowRanking" not in feedback_df.columns:
        return

    ranking_rows = []

    for _, row in feedback_df.iterrows():
        ranking = parse_workflow_ranking(row["workflowRanking"])

        for rank_index, workflow in enumerate(ranking, start=1):
            ranking_rows.append(
                {
                    "workflow": workflow,
                    "rank": rank_index,
                }
            )

    if not ranking_rows:
        return

    ranking_df = pd.DataFrame(ranking_rows)

    average_rank = (
        ranking_df.groupby("workflow")["rank"].mean().reindex(WORKFLOW_ORDER).dropna()
    )

    sorted_workflows = average_rank.sort_values(ascending=True).index.tolist()

    rank_counts = (
        ranking_df.groupby(["workflow", "rank"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted_workflows, fill_value=0)
    )

    rank_counts = rank_counts.rename(index=WORKFLOW_LABELS)
    rank_counts.columns = [f"Rank {int(column)}" for column in rank_counts.columns]

    average_rank_sorted = average_rank.reindex(sorted_workflows).rename(
        index=WORKFLOW_LABELS
    )

    rank_counts["Average rank"] = average_rank_sorted.round(2)

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
        "Distribution of final workflow rankings and average rank per workflow, sorted by average rank.",
    )


def plot_workflow(df, feedback_df):
    plot_total_workflow_usage_counts(df)
    plot_workflow_distribution(df)
    plot_workflow_transitions(df)
    plot_final_workflow_ranking(feedback_df)
