import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from scripts.config import TABLE_DIR, WORKFLOW_LABELS, EXPOSURE_LABELS, WORKFLOW_ORDER
from .utils import save_figure


ERROR_ROUND_INDEX = 5
POST_ERROR_ROUNDS = [6, 7]


def exposure_label(value):
    return EXPOSURE_LABELS.get(str(value), str(value))


def workflow_label(value):
    return WORKFLOW_LABELS.get(str(value), str(value))


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


def plot_error_exposure_overview(df: pd.DataFrame):
    """
    Shows how many participants were actually exposed to the AI error.

    A participant is error-exposed if they selected an AI-supported workflow
    in round 5.
    """

    participant_df = df[["participantId", "errorExposureGroup"]].drop_duplicates()

    if participant_df.empty:
        return

    summary = (
        participant_df["errorExposureGroup"]
        .value_counts()
        .rename_axis("errorExposureGroup")
        .reset_index(name="count")
    )

    summary["groupLabel"] = summary["errorExposureGroup"].map(exposure_label)

    summary.to_csv(TABLE_DIR / "error_exposure_overview.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(summary["groupLabel"], summary["count"])

    ax.set_title("Participant Error Exposure Overview")
    ax.set_xlabel("Exposure group")
    ax.set_ylabel("Number of participants")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, summary["count"].max() + 1)
    ax.bar_label(bars, padding=3)

    save_figure(
        fig,
        "21_error_exposure_overview",
        "Participant Error Exposure Overview",
        "Number of participants who were or were not exposed to the AI error in round 5.",
    )


def plot_round5_workflow_exposure(df: pd.DataFrame):
    """
    Shows which workflow participants selected in round 5 and whether that
    resulted in actual AI-error exposure.

    Round 5 is the first main round and the possible AI-error round.
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
        "22_round5_workflow_exposure",
        "Round-5 Workflow and Error Exposure",
        "Participants were exposed to the AI error only if their round-5 workflow used AI support.",
    )


def plot_post_error_workflow_choices_by_exposure(df: pd.DataFrame):
    """
    Shows exact workflow choices after the possible AI-error round.

    Error round = round 5.
    Post-error rounds = rounds 6–7.
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
        "23_post_error_workflow_choices_by_exposure",
        "Post-Error Workflow Choices by Error Exposure",
        "Distribution of workflow choices in rounds 6–7, split by whether participants were exposed to the AI error in round 5.",
    )


def plot_main_round_workflow_distribution_by_exposure(df: pd.DataFrame):
    """
    Shows workflow distribution across the full main phase.

    Main rounds = rounds 5–7.
    This includes the possible error round itself.
    """

    main_df = df[df["roundIndex"] >= ERROR_ROUND_INDEX].copy()

    if main_df.empty:
        return

    summary = (
        main_df.groupby(["errorExposureGroup", "workflow"])
        .size()
        .reset_index(name="count")
    )

    total_per_group = summary.groupby("errorExposureGroup")["count"].transform("sum")
    summary["percent"] = summary["count"] / total_per_group * 100
    summary["workflowLabel"] = summary["workflow"].map(workflow_label)
    summary["groupLabel"] = summary["errorExposureGroup"].map(exposure_label)

    summary.to_csv(
        TABLE_DIR / "main_round_workflow_distribution_by_exposure.csv",
        index=False,
        )

    pivot = (
        summary.pivot(index="groupLabel", columns="workflowLabel", values="percent")
        .fillna(0)
    )

    pivot = pivot[order_workflow_labels(pivot.columns)]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Workflow Distribution in Main Rounds by Error Exposure")
    ax.set_xlabel("Exposure group")
    ax.set_ylabel("Share of workflow choices in rounds 5–7 (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 100)

    save_figure(
        fig,
        "24_main_round_workflow_distribution_by_error_exposure",
        "Workflow Distribution by Error Exposure",
        "Distribution of selected workflows in rounds 5–7, split by error exposure group.",
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
        "frustration": "Frustration",
        "satisfactionResult": "Satisfaction",
        "aiPerformanceOverall": "AI performance",
    }

    for column, label in metrics.items():
        if column not in main_df.columns:
            continue

        plot_df = main_df.dropna(subset=[column]).copy()

        if plot_df.empty:
            continue

        summary = (
            plot_df.groupby(["errorExposureGroup", "roundIndex"])[column]
            .mean()
            .reset_index()
            .sort_values(["errorExposureGroup", "roundIndex"])
        )

        summary.to_csv(
            TABLE_DIR / f"{column}_by_exposure_and_main_round.csv",
            index=False,
            )

        fig, ax = plt.subplots(figsize=(7.4, 4.4))

        for group, group_df in summary.groupby("errorExposureGroup"):
            ax.plot(
                group_df["roundIndex"],
                group_df[column],
                marker="o",
                label=exposure_label(group),
            )

        ax.set_title(f"{label} over Main Rounds by Error Exposure")
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Mean {label.lower()} rating")
        ax.set_xticks(sorted(summary["roundIndex"].dropna().unique()))
        ax.legend(title="Exposure group")

        save_figure(
            fig,
            f"25_{column}_over_main_rounds_by_error_exposure",
            f"{label} over Main Rounds by Error Exposure",
            f"Mean {label.lower()} rating over rounds 5–7 for exposed and non-exposed participants.",
        )


def plot_post_error_experience_summary_by_exposure(df: pd.DataFrame):
    """
    Summarizes participant experience after the possible AI-error round.

    Error round = round 5.
    Post-error rounds = rounds 6–7.
    """

    post_df = df[df["roundIndex"] > ERROR_ROUND_INDEX].copy()

    if post_df.empty:
        return

    summary = (
        post_df.groupby("errorExposureGroup")
        .agg(
            meanFrustration=("frustration", "mean"),
            meanSatisfaction=("satisfactionResult", "mean"),
            meanAiPerformance=("aiPerformanceOverall", "mean"),
        )
        .reset_index()
    )

    summary["groupLabel"] = summary["errorExposureGroup"].map(exposure_label)
    summary.to_csv(TABLE_DIR / "post_error_experience_summary_by_exposure.csv", index=False)

    plot_df = summary.set_index("groupLabel")[
        ["meanFrustration", "meanSatisfaction", "meanAiPerformance"]
    ]

    plot_df = plot_df.rename(columns={
        "meanFrustration": "Frustration",
        "meanSatisfaction": "Satisfaction",
        "meanAiPerformance": "AI performance",
    })

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    plot_df.plot(kind="bar", ax=ax)

    ax.set_title("Post-Error Experience by Error Exposure")
    ax.set_xlabel("Exposure group")
    ax.set_ylabel("Mean rating in rounds 6–7")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Measure")

    save_figure(
        fig,
        "26_post_error_experience_by_error_exposure",
        "Post-Error Experience by Error Exposure",
        "Mean frustration, satisfaction, and AI performance in rounds 6–7 by error exposure group.",
    )


def generate_error_exposure_figures(df: pd.DataFrame):
    plot_error_exposure_overview(df)
    plot_round5_workflow_exposure(df)
    plot_post_error_workflow_choices_by_exposure(df)
    plot_main_round_workflow_distribution_by_exposure(df)
    plot_experience_over_main_rounds_by_exposure(df)
    plot_post_error_experience_summary_by_exposure(df)
