import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_ORDER, TABLE_DIR, WORKFLOW_LABELS
from scripts.dashboard_figures.utils import save_figure, workflow_label


def plot_composite_quality_over_rounds(df):
    slug = "11_composite_quality_over_rounds"
    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        metric = "meanOverallQuality"

    if metric not in df.columns or df[metric].dropna().empty:
        return

    summary = (
        df
        .dropna(subset=[metric, "roundIndex"])
        .groupby("roundIndex")[metric]
        .mean()
        .reset_index(name="meanQuality")
        .sort_values("roundIndex")
    )

    if summary.empty:
        return

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        summary["roundIndex"],
        summary["meanQuality"],
        marker="o",
    )

    ax.set_title("Composite Poem Quality over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean composite quality score (1–5)")
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(
        fig,
        slug,
        "Composite Quality over Rounds",
        "Mean externally rated composite poem quality score per round on a 1–5 scale, aggregated from Fluency, Theme Alignment, Meaningfulness, Poeticness, and Overall Quality scores.",
    )


def plot_composite_quality_by_workflow(df):
    slug = "12_composite_quality_by_workflow"
    metric = "qualityComposite"

    if metric not in df.columns or df[metric].dropna().empty:
        metric = "meanOverallQuality"

    if metric not in df.columns or df[metric].dropna().empty:
        return

    summary = (
        df
        .dropna(subset=[metric, "workflow"])
        .groupby("workflow")[metric]
        .mean()
        .reindex(WORKFLOW_ORDER)
        .dropna()
    )

    if summary.empty:
        return

    labeled_summary = summary.rename(index=WORKFLOW_LABELS)

    labeled_summary.to_csv(
        TABLE_DIR / f"{slug}.csv",
        header=["meanCompositeQuality"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    bars = ax.bar(
        labeled_summary.index,
        labeled_summary.values,
    )

    ax.bar_label(bars, padding=3, fmt="%.2f")

    ax.set_title("Composite Quality by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean composite quality score (1–5)")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(
        fig,
        slug,
        "Composite Quality by Workflow",
        "Mean externally rated composite poem quality score by workflow on a 1–5 scale, aggregated from Fluency, Theme Alignment, Meaningfulness, Poeticness, and Overall Quality scores.",
    )


def plot_quality_dimensions_by_workflow(df):
    slug = "13_quality_dimensions_by_workflow"

    dimension_columns = {
        "meanFluency": "Fluency",
        "meanThemeAlignment": "Theme alignment",
        "meanMeaningfulness": "Meaningfulness",
        "meanPoeticness": "Poeticness",
        "meanOverallQuality": "Overall quality",
    }

    available_columns = [
        column for column in dimension_columns
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


def plot_composite_quality_vs_time(df):
    slug = "14_composite_quality_vs_time_scatterplot"
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
    ]].to_csv(TABLE_DIR / f"{slug}.csv", index=False)

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

    ax.set_title("Output Composite Quality vs. Completion Time")
    ax.set_xlabel("Time used (minutes)")
    ax.set_ylabel("Mean composite quality score (1-5)")
    ax.legend(title="Workflow")

    save_figure(
        fig,
        slug,
        "Composite Quality vs Time Scatterplot",
        "Relationship between completion time and composite poem quality score on a 1–5 scale. Reported completion times include pauses recorded during the writing process.",
    )


def plot_composite_quality_efficiency_by_workflow(df):
    slug = "15_composite_quality_efficiency_by_workflow"
    metric = "qualityPerMinute"

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
        TABLE_DIR / f"{slug}.csv",
        header=["meanQualityPerMinute"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(summary.rename(index=WORKFLOW_LABELS).index, summary.values)

    ax.bar_label(bars, padding=3, fmt="%.2f")

    ax.set_title("Composite Quality Efficiency by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean composite quality score per minute")
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        slug,
        "Composite Quality Efficiency by Workflow",
        "Mean externally rated composite poem quality score on a 1-5 scale per minute by workflow.",
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
        topic = row["topic"] if "topic" in row and not pd.isna(row["topic"]) else "Unknown topic"

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
    best_slug = "16_best_poems_by_quality"
    worst_slug = "16b_worst_poems_by_quality"

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
        column for column in export_columns
        if column in plot_df.columns
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


def plot_constraint_fulfillment_over_rounds(df):
    slug = "17_constraint_fulfillment_over_rounds"

    if "constraintScore" not in df.columns or df["constraintScore"].dropna().empty:
        return

    plot_df = df.dropna(subset=["constraintScore", "roundIndex"]).copy()

    summary = (
        plot_df
        .groupby("roundIndex")["constraintScore"]
        .mean()
        .reset_index(name="meanConstraintScore")
        .sort_values("roundIndex")
    )

    if summary.empty:
        return

    summary.to_csv(
        TABLE_DIR / f"{slug}.csv",
        index=False,
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        summary["roundIndex"],
        summary["meanConstraintScore"],
        marker="o",
    )

    ax.axvline(5, linestyle="--", linewidth=1)

    ax.set_title("Constraint Fulfillment over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean constraint fulfillment (%)")
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
    ax.set_ylim(0, 100)

    save_figure(
        fig,
        slug,
        "Constraint Fulfillment over Rounds",
        "Mean constraint fulfillment percentage per round, with round 5 marked as the injected-error round.",
    )


def plot_constraint_rate_by_workflow(df):
    slug = "18_passed_constraint_rate_by_workflow"

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
        TABLE_DIR / f"{slug}.csv",
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
        slug,
        "Passed Constraint Rate by Workflow",
        "Percentage of rounds that passed the task constraints.",
    )


def plot_quality(df):
    plot_composite_quality_over_rounds(df)
    plot_composite_quality_by_workflow(df)
    plot_quality_dimensions_by_workflow(df)
    plot_composite_quality_vs_time(df)
    plot_composite_quality_efficiency_by_workflow(df)
    plot_best_and_worst_poems_by_quality(df)
    plot_constraint_fulfillment_over_rounds(df)
    plot_constraint_rate_by_workflow(df)
