from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import WORKFLOW_ORDER, TABLE_DIR, WORKFLOW_LABELS
from scripts.dashboard_figures.utils import save_figure, workflow_label


def plot_quality_over_rounds(df):
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

    summary.to_csv(TABLE_DIR / "quality_over_rounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        summary["roundIndex"],
        summary["meanQuality"],
        marker="o",
    )

    ax.set_title("Output Quality over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean quality score")
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(
        fig,
        "11_quality_over_rounds",
        "Output Quality over Rounds",
        "Mean externally rated output quality per round, based on available evaluator ratings.",
    )


def plot_quality_dimensions_by_workflow(df):
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

    summary.to_csv(TABLE_DIR / "quality_dimensions_by_workflow.csv")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    summary.plot(kind="bar", ax=ax)

    ax.set_title("Evaluator Rating Dimensions by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean evaluator rating")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Rating dimension", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(
        fig,
        "12_quality_dimensions_by_workflow",
        "Evaluator Rating Dimensions by Workflow",
        "Mean evaluator ratings by workflow, based on available evaluator-rated outputs.",
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
        "13_quality_vs_time_scatterplot",
        "Quality vs Time Scatterplot",
        "Relationship between completion time and externally rated output quality.",
    )


def plot_quality_efficiency_by_workflow(df):
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
        TABLE_DIR / "quality_efficiency_by_workflow.csv",
        header=["meanQualityPerMinute"],
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(summary.rename(index=WORKFLOW_LABELS).index, summary.values)

    ax.bar_label(bars, padding=3, fmt="%.2f")

    ax.set_title("Quality Efficiency by Workflow")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean quality per minute")
    ax.tick_params(axis="x", rotation=0)

    save_figure(
        fig,
        "14_quality_efficiency_by_workflow",
        "Quality Efficiency by Workflow",
        "Mean externally rated quality per minute by workflow.",
    )


def plot_constraint_fulfillment_over_rounds(df):
    if "constraintScore" in df.columns and not df["constraintScore"].dropna().empty:
        plot_df = df.dropna(subset=["constraintScore", "roundIndex"]).copy()

        summary = (
            plot_df
            .groupby("roundIndex")["constraintScore"]
            .mean()
            .reset_index(name="meanConstraintScore")
            .sort_values("roundIndex")
        )

        y_column = "meanConstraintScore"
        y_label = "Mean constraint score"
    elif {"constraintPassedCount", "constraintCount"}.issubset(df.columns):
        plot_df = df.dropna(subset=["constraintPassedCount", "constraintCount", "roundIndex"]).copy()
        plot_df = plot_df[plot_df["constraintCount"] > 0]

        if plot_df.empty:
            return

        plot_df["constraintRate"] = (
                plot_df["constraintPassedCount"] / plot_df["constraintCount"] * 100
        )

        summary = (
            plot_df
            .groupby("roundIndex")["constraintRate"]
            .mean()
            .reset_index(name="meanConstraintRate")
            .sort_values("roundIndex")
        )

        y_column = "meanConstraintRate"
        y_label = "Mean fulfilled constraints (%)"
    else:
        return

    if summary.empty:
        return

    summary.to_csv(TABLE_DIR / "constraint_fulfillment_over_rounds.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ax.plot(
        summary["roundIndex"],
        summary[y_column],
        marker="o",
    )

    ax.axvline(5, linestyle="--", linewidth=1)

    ax.set_title("Constraint Fulfillment over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))

    save_figure(
        fig,
        "15_constraint_fulfillment_over_rounds",
        "Constraint Fulfillment over Rounds",
        "Mean constraint fulfillment per round, with round 5 marked as the injected-error round.",
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
        "16_passed_constraint_rate_by_workflow",
        "Passed Constraint Rate by Workflow",
        "Percentage of rounds that passed the task constraints.",
    )


def plot_quality(df):
    plot_quality_over_rounds(df)
    plot_quality_dimensions_by_workflow(df)
    plot_quality_vs_time(df)
    plot_quality_efficiency_by_workflow(df)
    plot_constraint_fulfillment_over_rounds(df)
    plot_constraint_rate_by_workflow(df)
