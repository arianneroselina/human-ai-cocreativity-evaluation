from matplotlib import pyplot as plt

from scripts.config import WORKFLOW_ORDER, TABLE_DIR, WORKFLOW_LABELS
from scripts.dashboard_figures.utils import save_figure


def plot_satisfaction_by_workflow(df):
    column = "satisfactionResult"

    if column not in df.columns:
        return

    summary = (
        df
        .dropna(subset=[column])
        .groupby("workflow")[column]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna()
        .rename(index=WORKFLOW_LABELS)
    )

    if summary.empty:
        return

    summary.to_csv(
        TABLE_DIR / "satisfaction_by_workflow.csv",
        header=["meanSatisfaction"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(summary.index, summary.values)

    ax.bar_label(bars, padding=3, fmt="%.2f")

    ax.set_title("Participant Satisfaction by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean satisfaction rating (1–5)")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        "21_satisfaction_by_workflow",
        "Participant Satisfaction by Workflow",
        "Mean participant-reported satisfaction ratings on a 1–5 scale after each writing round by workflow.",
    )


def plot_ai_experience_over_rounds(df):
    ai_metrics = {
        "aiUnderstanding": "AI understanding",
        "aiCollaboration": "AI collaboration",
        "aiCreativitySupport": "AI creativity support",
        "aiPerformanceOverall": "AI performance",
    }

    available_metrics = [
        column for column in ai_metrics
        if column in df.columns and not df[column].dropna().empty
    ]

    if not available_metrics:
        return

    ai_df = df[df["workflow"] != "human"].copy()

    if ai_df.empty:
        return

    summary = (
        ai_df
        .groupby("roundIndex")[available_metrics]
        .mean()
        .reset_index()
        .sort_values("roundIndex")
    )

    summary = summary.rename(columns=ai_metrics)
    summary.to_csv(TABLE_DIR / "ai_experience_over_rounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for metric_label in ai_metrics.values():
        if metric_label not in summary.columns:
            continue

        ax.plot(
            summary["roundIndex"],
            summary[metric_label],
            marker="o",
            label=metric_label,
        )

    ax.set_title("AI Collaboration Experience over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean rating (1–5)")
    ax.set_ylim(0, 5)
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
    ax.legend(title="AI-related measure")

    save_figure(
        fig,
        "22_ai_experience_over_rounds",
        "AI Collaboration Experience over Rounds",
        "Mean participant ratings on a 1–5 scale for AI understanding, collaboration quality, creativity support, and overall AI performance across AI-supported rounds.",
    )


def plot_tlx_subscale_ratings_by_workflow(df):
    load_columns = {
        "mentalDemand": "Mental demand",
        "physicalDemand": "Physical demand",
        "temporalDemand": "Temporal demand",
        "performance": "Performance (lower = better)",
        "effort": "Effort",
        "frustration": "Frustration",
    }

    available_columns = [
        column for column in load_columns
        if column in df.columns and not df[column].dropna().empty
    ]

    if not available_columns:
        return

    summary = (
        df
        .groupby("workflow")[available_columns]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .rename(index=WORKFLOW_LABELS, columns=load_columns)
    )

    if summary.empty:
        return

    summary.to_csv(TABLE_DIR / "task_load_by_workflow.csv")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    summary.plot(kind="bar", ax=ax)

    ax.set_title("NASA-TLX Subscale Ratings by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean NASA-TLX subscale rating (1–21)")
    ax.set_ylim(0, 21)
    ax.tick_params(axis="x", rotation=0)

    ax.legend(
        title="NASA-TLX subscale",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        "23_tlx_subscale_ratings_by_workflow",
        "NASA-TLX Subscale Ratings by Workflow",
        "Mean NASA-TLX subscale ratings on a 1–21 scale across workflows.",
    )


def plot_experience(df):
    plot_satisfaction_by_workflow(df)
    plot_ai_experience_over_rounds(df)
    plot_tlx_subscale_ratings_by_workflow(df)
