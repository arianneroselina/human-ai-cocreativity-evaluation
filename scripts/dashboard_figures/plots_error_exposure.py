import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from scripts.config import TABLE_DIR, EXPOSURE_LABELS, WORKFLOW_ORDER
from .utils import save_figure, workflow_label


ERROR_ROUND_INDEX = 5


def exposure_label(value):
    return EXPOSURE_LABELS.get(str(value), str(value))


def order_workflow_labels(labels):
    ordered_labels = [
        workflow_label(workflow)
        for workflow in WORKFLOW_ORDER
        if workflow_label(workflow) in labels
    ]

    remaining_labels = [
        label for label in labels
        if label not in ordered_labels
    ]

    return ordered_labels + remaining_labels


def plot_round5_workflow_exposure(df: pd.DataFrame):
    """
    Shows which workflow participants selected in round 5 and whether that
    resulted in actual AI-error exposure.
    """

    round5_df = df[df["roundIndex"] == ERROR_ROUND_INDEX].copy()

    if round5_df.empty:
        return

    summary = (
        round5_df.groupby(["workflow", "errorExposureGroup"])
        .size()
        .reset_index(name="count")
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)
    summary["groupLabel"] = summary["errorExposureGroup"].map(exposure_label)

    summary.to_csv(TABLE_DIR / "round5_workflow_exposure.csv", index=False)

    pivot = (
        summary.pivot(index="workflowLabel", columns="groupLabel", values="count")
        .fillna(0)
    )

    pivot = pivot.reindex(order_workflow_labels(pivot.index))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Round-5 Workflow and Actual Error Exposure")
    ax.set_xlabel("Workflow selected in round 5")
    ax.set_ylabel("Number of participants")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Exposure group")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=8)

    save_figure(
        fig,
        "31_round5_workflow_exposure",
        "Round-5 Workflow and Error Exposure",
        "Participants were exposed to the AI error only if their round-5 workflow used AI support.",
    )


def extract_line_count_error(requirement_results):
    if pd.isna(requirement_results):
        return None

    try:
        results = json.loads(requirement_results)
    except json.JSONDecodeError:
        return None

    for item in results:
        rule_id = str(item.get("id", ""))

        if rule_id.startswith("lines-"):
            return not bool(item.get("passed"))

    return None


def plot_line_count_error_by_round_ai_workflows(df):
    """
    Shows whether the line-count constraint error increases in round 5.
    """

    if "requirementResults" not in df.columns:
        return

    ai_workflows = ["ai", "human_ai", "ai_human"]

    plot_df = df[df["workflow"].isin(ai_workflows)].copy()
    plot_df["lineCountError"] = plot_df["requirementResults"].apply(
        extract_line_count_error
    )

    plot_df = plot_df.dropna(subset=["lineCountError", "roundIndex", "workflow"])

    if plot_df.empty:
        return

    summary = (
        plot_df
        .groupby(["roundIndex", "workflow"])["lineCountError"]
        .mean()
        .mul(100)
        .reset_index(name="lineCountErrorRatePercent")
    )

    summary.to_csv(TABLE_DIR / "line_count_error_by_round_ai_workflows.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for workflow in ai_workflows:
        workflow_df = summary[summary["workflow"] == workflow]

        if workflow_df.empty:
            continue

        ax.plot(
            workflow_df["roundIndex"],
            workflow_df["lineCountErrorRatePercent"],
            marker="o",
            label=workflow_label(workflow),
        )

    ax.axvline(ERROR_ROUND_INDEX, linestyle="--", linewidth=1)
    ax.text(
        ERROR_ROUND_INDEX + 0.05,
        90,
        "Injected AI line-count error",
        fontsize=9,
        )

    ax.set_title("Line-Count Error Rate over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Line-count error rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="AI-supported workflow")

    save_figure(
        fig,
        "32_line_count_error_by_round_ai_workflows",
        "Line-Count Error by Round",
        "Line-count constraint error rate across rounds for AI-supported workflows.",
    )


def plot_post_error_workflow_choices_by_exposure(df: pd.DataFrame):
    """
    Shows exact workflow choices after the possible AI-error round.
    """

    choice_df = df[df["roundIndex"] > ERROR_ROUND_INDEX].copy()

    if choice_df.empty:
        return

    summary = (
        choice_df
        .groupby(["errorExposureGroup", "workflow"])
        .size()
        .reset_index(name="count")
    )

    summary["groupTotal"] = summary.groupby("errorExposureGroup")["count"].transform("sum")
    summary["percent"] = summary["count"] / summary["groupTotal"] * 100

    summary["groupLabel"] = summary["errorExposureGroup"].map(exposure_label)
    summary["workflowLabel"] = summary["workflow"].map(workflow_label)

    summary.to_csv(TABLE_DIR / "post_error_workflow_choices_by_exposure.csv", index=False)

    pivot = (
        summary
        .pivot(index="groupLabel", columns="workflowLabel", values="percent")
        .fillna(0)
    )

    pivot = pivot[order_workflow_labels(pivot.columns)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Post-Error Workflow Choices by Error Exposure")
    ax.set_xlabel("Exposure group")
    ax.set_ylabel("Share of workflow choices in rounds 6–7 (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Chosen workflow")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8)

    save_figure(
        fig,
        "33_post_error_workflow_choices_by_exposure",
        "Post-Error Workflow Choices by Error Exposure",
        "Distribution of workflow choices in rounds 6–7, split by whether participants were exposed to the AI error in round 5.",
    )


def plot_experience_over_main_rounds_by_exposure(df: pd.DataFrame):
    """
    Shows participant experience over the main rounds, split by exposure group.

    Main rounds = rounds 5–7.
    """

    main_df = df[df["roundIndex"] >= ERROR_ROUND_INDEX].copy()

    if main_df.empty:
        return

    metrics = {
        "satisfactionResult": "Satisfaction",
        "frustration": "Frustration",
        "aiPerformanceOverall": "AI performance",
    }

    available_metrics = [
        column for column in metrics
        if column in main_df.columns and not main_df[column].dropna().empty
    ]

    if not available_metrics:
        return

    summary = (
        main_df
        .groupby(["errorExposureGroup", "roundIndex"])[available_metrics]
        .mean()
        .reset_index()
        .sort_values(["errorExposureGroup", "roundIndex"])
    )

    summary.to_csv(
        TABLE_DIR / "experience_by_exposure_and_main_round.csv",
        index=False,
        )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for group, group_df in summary.groupby("errorExposureGroup"):
        group_label = exposure_label(group)

        for column in available_metrics:
            ax.plot(
                group_df["roundIndex"],
                group_df[column],
                marker="o",
                label=f"{group_label} - {metrics[column]}",
            )

    ax.set_title("Experience over Main Rounds by Error Exposure")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean rating")
    ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
    ax.legend(title="Group and measure", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_figure(
        fig,
        "34_experience_over_main_rounds_by_error_exposure",
        "Experience over Main Rounds by Error Exposure",
        "Mean satisfaction, frustration, and AI performance over rounds 5–7, split by error exposure group.",
    )


def plot_error_exposure(df: pd.DataFrame):
    plot_round5_workflow_exposure(df)
    plot_line_count_error_by_round_ai_workflows(df)
    plot_post_error_workflow_choices_by_exposure(df)
    plot_experience_over_main_rounds_by_exposure(df)
