import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_ORDER, TABLE_DIR, WORKFLOW_LABELS, ERROR_ROUND_INDEX
from scripts.utils import (
    save_figure,
    workflow_label,
    shade_main_rounds,
    annotate_injected_error_round,
)


def get_quality_metric(df):
    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        metric = "meanOverallQuality"

    if metric not in df.columns or df[metric].dropna().empty:
        return None

    return metric


def plot_composite_quality_by_round_and_workflow(df):
    """
    This figure shows the interaction between round and workflow, because
    composite quality should not be interpreted by round or workflow alone.
    """
    slug = "11_composite_quality_by_round_and_workflow"

    metric = get_quality_metric(df)
    if metric is None:
        return

    required_columns = {metric, "roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    plot_df = df.dropna(subset=[metric, "roundIndex", "workflow"]).copy()

    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(["roundIndex", "workflow"])[metric]
        .agg(
            meanQuality="mean",
            count="count",
        )
        .reset_index()
        .sort_values(["roundIndex", "workflow"])
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    pivot = summary.pivot(
        index="roundIndex", columns="workflow", values="meanQuality"
    ).reindex(columns=WORKFLOW_ORDER)

    pivot = pivot.rename(columns=WORKFLOW_LABELS)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    shade_main_rounds(ax)

    for workflow in WORKFLOW_ORDER:
        workflow_label_text = WORKFLOW_LABELS.get(workflow, workflow)

        if workflow_label_text not in pivot.columns:
            continue

        workflow_series = pivot[workflow_label_text].dropna()

        if workflow_series.empty:
            continue

        ax.plot(
            workflow_series.index,
            workflow_series.values,
            marker="o",
            label=workflow_label_text,
        )

    annotate_injected_error_round(ax, ERROR_ROUND_INDEX, y_top=5, text_y=5.25)

    ax.set_title("Composite Quality by Round and Workflow")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean composite quality score (1–5)")
    ax.set_xticks(sorted(plot_df["roundIndex"].dropna().unique()))
    ax.set_ylim(0, 5.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Composite Quality by Round and Workflow",
        "Mean externally rated composite poem quality score by round and workflow. This figure combines round effects and workflow effects instead of interpreting them separately.",
    )


def plot_quality_dimensions_by_workflow(df):
    slug = "12_quality_dimensions_by_workflow"

    dimension_columns = {
        "meanFluency": "Fluency",
        "meanThemeAlignment": "Theme alignment",
        "meanMeaningfulness": "Meaningfulness",
        "meanPoeticness": "Poeticness",
        "meanOverallQuality": "Overall quality",
    }

    available_columns = [
        column
        for column in dimension_columns
        if column in df.columns and not df[column].dropna().empty
    ]

    if not available_columns:
        return

    summary = (
        df.groupby("workflow")[available_columns]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .rename(index=WORKFLOW_LABELS, columns=dimension_columns)
    )

    if summary.empty:
        return

    summary.to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    summary.plot(kind="bar", ax=ax)

    ax.set_title("Poem Evaluation Dimensions by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean rating score (1–5)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Evaluation dimension", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(
        fig,
        slug,
        "Poem Evaluation Dimensions by Workflow",
        "Mean poem evaluation scores by workflow on a 1–5 scale, across the dimensions Fluency, Theme Alignment, Meaningfulness, Poeticness, and Overall Quality.",
    )


def plot_quality_time_efficiency_by_workflow(df):
    """
    Merges the previous quality-vs-time scatterplot and quality-efficiency figure.

    Each point is one workflow:
    - x-axis: mean completion time
    - y-axis: mean composite quality
    - annotation: quality per minute and sample size
    """
    slug = "13_quality_time_efficiency_by_workflow"

    metric = get_quality_metric(df)
    if metric is None:
        return

    time_column = "effectiveTimeMinutes"

    if time_column not in df.columns:
        if "timeMs" not in df.columns:
            return

        df = df.copy()
        df[time_column] = df["timeMs"] / 60000

    required_columns = {metric, time_column, "workflow"}

    if not required_columns.issubset(df.columns):
        return

    plot_df = df.dropna(subset=[metric, time_column, "workflow"]).copy()

    if plot_df.empty:
        return

    summary = (
        plot_df.groupby("workflow")
        .agg(
            meanQuality=(metric, "mean"),
            meanTimeMinutes=(time_column, "mean"),
            count=(metric, "count"),
        )
        .reindex(WORKFLOW_ORDER)
        .dropna(subset=["meanQuality", "meanTimeMinutes"])
    )

    if summary.empty:
        return

    summary["qualityPerMinute"] = summary["meanQuality"] / summary["meanTimeMinutes"]

    summary = summary.rename(index=WORKFLOW_LABELS)

    summary.to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.scatter(
        summary["meanTimeMinutes"],
        summary["meanQuality"],
    )

    for workflow_label_text, row in summary.iterrows():
        ax.annotate(
            f"{workflow_label_text}\n"
            f"Quality/min: {row['qualityPerMinute']:.2f}\n"
            f"n={int(row['count'])}",
            xy=(row["meanTimeMinutes"], row["meanQuality"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Composite Quality, Time, and Efficiency by Workflow")
    ax.set_xlabel("Mean completion time (minutes)")
    ax.set_ylabel("Mean composite quality score (1–5)")
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(
        fig,
        slug,
        "Composite Quality, Time, and Efficiency by Workflow",
        "Workflow-level comparison of mean completion time, mean externally rated composite quality, and quality per minute.",
    )


def _wrap_poem_text(text, max_line_length=80):
    lines = []

    for original_line in str(text).splitlines():
        line = original_line.strip()

        if not line:
            lines.append("")
            continue

        while len(line) > max_line_length:
            split_index = line.rfind(" ", 0, max_line_length)

            if split_index == -1:
                split_index = max_line_length

            lines.append(line[:split_index].strip())
            line = line[split_index:].strip()

        lines.append(line)

    return "\n".join(lines)


def _plot_poem_text_figure(poem_df, slug, title, description, metric):
    if poem_df.empty:
        return

    poem_df = poem_df.copy()
    poem_df["workflowLabel"] = poem_df["workflow"].map(workflow_label)

    fig_height = max(4.8, 3.2 * len(poem_df))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    y_position = 0.95
    card_height = 0.86 / len(poem_df)

    for index, row in poem_df.reset_index(drop=True).iterrows():
        topic = (
            row["topic"]
            if "topic" in row and not pd.isna(row["topic"])
            else "Unknown topic"
        )

        header = (
            f"Poem {index + 1}   |   "
            f"Round {int(row['roundIndex'])}   |   "
            f"{row['workflowLabel']}   |   "
            f"Mean quality: {row[metric]:.2f}"
        )

        wrapped_poem = _wrap_poem_text(row["text"], max_line_length=72)

        card_top = y_position
        card_bottom = y_position - card_height + 0.03

        ax.text(
            0.03,
            card_top,
            header,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#f2f2f2",
                "edgecolor": "#d0d0d0",
                "linewidth": 1,
            },
        )

        ax.text(
            0.03,
            card_top - 0.075,
            f"Topic: {topic}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            style="italic",
        )

        ax.text(
            0.03,
            card_top - 0.14,
            wrapped_poem,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#ffffff",
                "edgecolor": "#d9d9d9",
                "linewidth": 1,
            },
        )

        y_position = card_bottom

    save_figure(
        fig,
        slug,
        title,
        description,
    )


def plot_best_and_worst_poems_by_quality(df):
    best_slug = "14_best_poems_by_quality"
    worst_slug = "14b_worst_poems_by_quality"

    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        metric = "meanOverallQuality"

    required_columns = [metric, "text", "workflow", "roundIndex"]

    if any(column not in df.columns for column in required_columns):
        return

    plot_df = df.dropna(subset=[metric, "text"]).copy()

    if plot_df.empty:
        return

    max_quality = plot_df[metric].max()
    min_quality = plot_df[metric].min()

    best_poems = plot_df[plot_df[metric] == max_quality].copy()
    worst_poems = plot_df[plot_df[metric] == min_quality].copy()

    export_columns = [
        "roundId",
        "roundIndex",
        "workflow",
        "topic",
        "text",
        metric,
    ]

    available_columns = [
        column for column in export_columns if column in plot_df.columns
    ]

    best_poems[available_columns].to_csv(
        TABLE_DIR / f"{best_slug}.csv",
        index=False,
    )

    worst_poems[available_columns].to_csv(
        TABLE_DIR / f"{worst_slug}.csv",
        index=False,
    )

    _plot_poem_text_figure(
        best_poems,
        best_slug,
        "Best Rated Poem(s)",
        "Poem text for all outputs with the highest mean quality score.",
        metric,
    )

    _plot_poem_text_figure(
        worst_poems,
        worst_slug,
        "Worst Rated Poem(s)",
        "Poem text for all outputs with the lowest mean quality score.",
        metric,
    )


def plot_quality(df):
    plot_composite_quality_by_round_and_workflow(df)
    plot_quality_dimensions_by_workflow(df)
    plot_quality_time_efficiency_by_workflow(df)
    plot_best_and_worst_poems_by_quality(df)
