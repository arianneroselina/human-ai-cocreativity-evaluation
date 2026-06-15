import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import ERROR_ROUND_INDEX, TABLE_DIR, WORKFLOW_LABELS, WORKFLOW_ORDER
from scripts.utils import (
    is_ai_supported_row,
    save_figure,
    shade_main_rounds,
    workflow_label,
    annotate_injected_error_round,
)


def drop_duplicate_participant_rounds(df: pd.DataFrame) -> pd.DataFrame:
    if "participantId" not in df.columns:
        return df

    return df.drop_duplicates(
        subset=["participantId", "roundIndex"],
        keep="first",
    )


def plot_satisfaction_by_round_and_workflow(df):
    """
    Shows participant satisfaction by both round and workflow.

    Satisfaction should not be interpreted only by workflow or only by round,
    because workflow choice and round progression may interact.
    """
    slug = "21_satisfaction_by_round_and_workflow"
    column = "satisfactionResult"

    required_columns = {column, "roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    plot_df = df.dropna(subset=[column, "roundIndex", "workflow"]).copy()
    plot_df = drop_duplicate_participant_rounds(plot_df)

    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df["roundIndex"] = pd.to_numeric(plot_df["roundIndex"], errors="coerce")

    plot_df = plot_df.dropna(subset=[column, "roundIndex"])

    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(["roundIndex", "workflow"])[column]
        .agg(
            meanSatisfaction="mean",
            count="count",
        )
        .reset_index()
        .sort_values(["roundIndex", "workflow"])
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)
    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    pivot = summary.pivot(
        index="roundIndex",
        columns="workflow",
        values="meanSatisfaction",
    ).reindex(columns=WORKFLOW_ORDER)

    if pivot.dropna(how="all").empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    shade_main_rounds(ax, label="Main rounds (5–7)", label_y=0.03)

    for workflow in WORKFLOW_ORDER:
        if workflow not in pivot.columns:
            continue

        workflow_series = pivot[workflow].dropna()

        if workflow_series.empty:
            continue

        ax.plot(
            workflow_series.index,
            workflow_series.values,
            marker="o",
            label=workflow_label(workflow),
            zorder=2,
        )

    annotate_injected_error_round(ax, ERROR_ROUND_INDEX, y_top=5.0, text_y=5.25)

    ax.set_title("Participant Satisfaction by Round and Workflow")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean satisfaction rating (1–5)")
    ax.set_ylim(0, 5.5)
    ax.set_xticks(sorted(plot_df["roundIndex"].dropna().unique()))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Participant Satisfaction by Round and Workflow",
        "Mean participant-reported satisfaction ratings on a 1–5 scale by round and workflow.",
    )


def plot_ai_experience_by_round_and_workflow(df):
    """
    Shows AI-related experience ratings by round and AI-supported workflow.

    To keep the figure readable, each AI-related rating dimension is shown
    in a separate subplot, while workflow is represented as lines.
    """
    slug = "22_ai_experience_by_round_and_workflow"

    ai_metrics = {
        "aiUnderstanding": "AI understanding",
        "aiCollaboration": "AI collaboration",
        "aiCreativitySupport": "AI creativity support",
        "aiPerformanceOverall": "AI performance",
    }

    required_columns = {"roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    available_metrics = [
        column
        for column in ai_metrics
        if column in df.columns and not df[column].dropna().empty
    ]

    if not available_metrics:
        return

    plot_df = df.dropna(subset=["roundIndex", "workflow"]).copy()
    plot_df = drop_duplicate_participant_rounds(plot_df)

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["roundIndex"])

    if plot_df.empty:
        return

    plot_df = plot_df[plot_df.apply(is_ai_supported_row, axis=1)].copy()

    if plot_df.empty:
        return

    for metric in available_metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")

    summary = (
        plot_df.groupby(["roundIndex", "workflow"])[available_metrics]
        .mean()
        .reset_index()
        .sort_values(["roundIndex", "workflow"])
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)
    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.2, 7.2),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    ai_workflows_in_order = [
        workflow for workflow in WORKFLOW_ORDER if workflow != "human"
    ]

    for ax, metric in zip(axes, available_metrics):
        metric_label = ai_metrics[metric]

        pivot = summary.pivot(
            index="roundIndex",
            columns="workflow",
            values=metric,
        ).reindex(columns=ai_workflows_in_order)

        shade_main_rounds(ax, label="", label_y=0.03)

        for workflow in ai_workflows_in_order:
            if workflow not in pivot.columns:
                continue

            workflow_series = pivot[workflow].dropna()

            if workflow_series.empty:
                continue

            ax.plot(
                workflow_series.index,
                workflow_series.values,
                marker="o",
                label=workflow_label(workflow),
                zorder=2,
            )

        ax.axvline(
            ERROR_ROUND_INDEX,
            linestyle="--",
            linewidth=1,
            zorder=3,
        )

        ax.set_title(metric_label)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean rating (1–5)")
        ax.set_ylim(0, 5.5)
        ax.set_xticks(sorted(plot_df["roundIndex"].dropna().unique()))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for ax in axes[len(available_metrics) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        title="AI-supported workflow",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    fig.suptitle(
        "AI Collaboration Experience by Round and Workflow",
        fontsize=14,
    )

    fig.text(
        0.48,
        0.02,
        "Shaded area marks the main rounds (5–7); dashed line marks the injected AI-error round.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    fig.tight_layout(rect=[0, 0.04, 0.86, 0.94])

    save_figure(
        fig,
        slug,
        "AI Collaboration Experience by Round and Workflow",
        "Mean participant ratings on a 1–5 scale for AI understanding, collaboration quality, creativity support, and overall AI performance, separated by round and AI-supported workflow.",
    )


def plot_tlx_subscale_ratings_by_workflow(df):
    slug = "23_tlx_subscale_ratings_by_workflow"

    load_columns = {
        "mentalDemand": "Mental demand",
        "physicalDemand": "Physical demand",
        "temporalDemand": "Temporal demand",
        "performance": "Performance (lower = better)",
        "effort": "Effort",
        "frustration": "Frustration",
    }

    available_columns = [
        column
        for column in load_columns
        if column in df.columns and not df[column].dropna().empty
    ]

    if not available_columns:
        return

    summary = (
        df.groupby("workflow")[available_columns]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .rename(index=WORKFLOW_LABELS, columns=load_columns)
    )

    if summary.empty:
        return

    summary.to_csv(TABLE_DIR / f"{slug}.csv")

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
        slug,
        "NASA-TLX Subscale Ratings by Workflow",
        "Mean NASA-TLX subscale ratings on a 1–21 scale across workflows.",
    )


def plot_experience(df):
    plot_satisfaction_by_round_and_workflow(df)
    plot_ai_experience_by_round_and_workflow(df)
    plot_tlx_subscale_ratings_by_workflow(df)
