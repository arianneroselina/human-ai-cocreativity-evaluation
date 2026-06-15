import json

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    ERROR_ROUND_INDEX,
    TABLE_DIR,
    WORKFLOW_ORDER,
)
from scripts.utils import (
    is_ai_supported_row,
    parse_bool_or_none,
    save_figure,
    workflow_label,
    shade_main_rounds,
    shade_main_rounds_for_bar_axis,
    annotate_injected_error_round,
)


def drop_duplicate_participant_rounds(df: pd.DataFrame) -> pd.DataFrame:
    if "participantId" not in df.columns:
        return df

    return df.drop_duplicates(
        subset=["participantId", "roundIndex"],
        keep="first",
    )


def to_passed_numeric(value):
    parsed = parse_bool_or_none(value)

    if parsed is True:
        return 1

    if parsed is False:
        return 0

    return pd.NA


def plot_constraint_fulfillment_by_round_and_workflow(df):
    """
    Merges the previous:
    - constraint fulfillment over rounds
    - passed constraint rate by workflow

    This figure shows constraint success by round and workflow together.
    """
    slug = "16_constraint_fulfillment_by_round_and_workflow"

    required_columns = {"roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    has_passed = "passed" in df.columns and not df["passed"].dropna().empty
    has_constraint_score = (
        "constraintScore" in df.columns and not df["constraintScore"].dropna().empty
    )

    if not has_passed and not has_constraint_score:
        return

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["roundIndex", "workflow"])
    plot_df = drop_duplicate_participant_rounds(plot_df)

    if plot_df.empty:
        return

    plot_df["roundIndex"] = pd.to_numeric(
        plot_df["roundIndex"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(subset=["roundIndex"])

    if plot_df.empty:
        return

    group_columns = ["roundIndex", "workflow"]

    summary = plot_df.groupby(group_columns).size().reset_index(name="roundCount")

    if has_passed:
        plot_df["passedNumeric"] = plot_df["passed"].apply(to_passed_numeric)

        pass_summary = (
            plot_df.dropna(subset=["passedNumeric"])
            .groupby(group_columns)["passedNumeric"]
            .mean()
            .mul(100)
            .reset_index(name="passedRatePercent")
        )

        summary = summary.merge(
            pass_summary,
            on=group_columns,
            how="left",
        )

    if has_constraint_score:
        plot_df["constraintScore"] = pd.to_numeric(
            plot_df["constraintScore"],
            errors="coerce",
        )

        score_summary = (
            plot_df.dropna(subset=["constraintScore"])
            .groupby(group_columns)["constraintScore"]
            .mean()
            .reset_index(name="meanConstraintScore")
        )

        summary = summary.merge(
            score_summary,
            on=group_columns,
            how="left",
        )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    if (
        "passedRatePercent" in summary.columns
        and not summary["passedRatePercent"].dropna().empty
    ):
        plot_metric = "passedRatePercent"
        y_label = "Passed constraint rate (%)"
        description_metric = "passed constraint rate"
    else:
        plot_metric = "meanConstraintScore"
        y_label = "Mean constraint fulfillment (%)"
        description_metric = "mean constraint fulfillment score"

    pivot = summary.pivot(
        index="roundIndex", columns="workflow", values=plot_metric
    ).reindex(columns=WORKFLOW_ORDER)

    if pivot.dropna(how="all").empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    shade_main_rounds(ax)

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
        )

    annotate_injected_error_round(ax, ERROR_ROUND_INDEX, y_top=90, text_y=95)

    ax.set_title("Constraint Fulfillment by Round and Workflow")
    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.set_xticks(sorted(plot_df["roundIndex"].dropna().unique()))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Constraint Fulfillment by Round and Workflow",
        f"Constraint fulfillment shown as {description_metric} by round and workflow, with round 5 marked as the injected-error round.",
    )


def classify_constraint_failure_context(row):
    """
    Classifies the context in which a submitted round failed its constraints.

    Important:
    - errorExposed is participant-level, not round-level.
    - Injected AI error exposure is only counted in round 5.
    - The injected-error category is only used if the submitted poem failed constraints.
    - AI-supported non-injected constraint failure means that the final submitted poem
      failed a constraint in an AI-supported workflow outside the injected-error round.
    """
    passed = parse_bool_or_none(row.get("passed"))
    has_constraint_error = passed is False

    round_index = row.get("roundIndex")
    error_exposed = parse_bool_or_none(row.get("errorExposed")) is True
    ai_supported = is_ai_supported_row(row)

    try:
        round_index = int(round_index)
    except (TypeError, ValueError):
        round_index = None

    is_injected_error_exposure_round = (
        round_index == ERROR_ROUND_INDEX and error_exposed
    )

    if has_constraint_error and is_injected_error_exposure_round:
        return "Injected AI error round constraint failure"

    if has_constraint_error and ai_supported:
        return "AI-supported non-injected constraint failure"

    if has_constraint_error and not ai_supported:
        return "Human-only constraint failure"

    return "No detected constraint failure"


def plot_constraint_failure_context_by_round(df: pd.DataFrame):
    """
    Shows whether detected constraint failures occurred in human-only workflows,
    AI-supported workflows, or during the injected AI-error exposure round.
    """
    slug = "17_constraint_failure_context_by_round"

    required_columns = {"roundIndex", "workflow", "passed"}

    if not required_columns.issubset(df.columns):
        return

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["roundIndex", "workflow"])
    plot_df = drop_duplicate_participant_rounds(plot_df)

    if plot_df.empty:
        return

    plot_df["constraintFailureContext"] = plot_df.apply(
        classify_constraint_failure_context,
        axis=1,
    )

    category_order = [
        "No detected constraint failure",
        "Human-only constraint failure",
        "AI-supported non-injected constraint failure",
        "Injected AI error round constraint failure",
    ]

    counts = (
        plot_df.groupby(["roundIndex", "constraintFailureContext"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=category_order, fill_value=0)
        .sort_index()
    )

    if counts.empty:
        return

    percentages = counts.div(counts.sum(axis=1), axis=0).mul(100)

    counts.to_csv(TABLE_DIR / f"{slug}_counts.csv")
    percentages.round(2).to_csv(TABLE_DIR / f"{slug}_percentages.csv")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    percentages.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        zorder=2,
    )

    shade_main_rounds_for_bar_axis(
        ax,
        round_indices=counts.index,
        label="Main rounds (5–7)",
    )

    round_indices = list(counts.index)

    if ERROR_ROUND_INDEX in round_indices:
        error_round_position = round_indices.index(ERROR_ROUND_INDEX)
        annotate_injected_error_round(ax, error_round_position, y_top=100, text_y=105)

    ax.set_title("Constraint Failure Context over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Share of submitted rounds (%)")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=0)

    ax.legend(
        title="Constraint failure context",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Constraint Failure Context over Rounds",
        "Detected constraint failures separated by human-only workflows, AI-supported workflows, and injected AI-error round constraint failures.",
    )


def extract_line_count_error(requirement_results):
    if pd.isna(requirement_results):
        return None

    try:
        results = json.loads(requirement_results)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(results, list):
        return None

    for item in results:
        rule_id = str(item.get("id", ""))

        if not rule_id.startswith("lines-"):
            continue

        passed = parse_bool_or_none(item.get("passed"))

        if passed is None:
            return None

        return not passed

    return None


def plot_line_count_error_by_round_ai_workflows(df):
    """
    Figure 18.

    Shows whether the line-count constraint error increases in round 5
    for AI-supported workflows.
    """
    slug = "18_line_count_error_by_round_ai_workflows"

    required_columns = {"requirementResults", "roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["roundIndex", "workflow"])
    plot_df = drop_duplicate_participant_rounds(plot_df)

    plot_df = plot_df[plot_df.apply(is_ai_supported_row, axis=1)].copy()

    if plot_df.empty:
        return

    plot_df["lineCountError"] = plot_df["requirementResults"].apply(
        extract_line_count_error,
    )

    plot_df = plot_df.dropna(subset=["lineCountError", "roundIndex", "workflow"])

    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(["roundIndex", "workflow"])["lineCountError"]
        .mean()
        .mul(100)
        .reset_index(name="lineCountErrorRatePercent")
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    shade_main_rounds(ax)

    workflows_in_order = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow in summary["workflow"].unique()
    ]

    for workflow in workflows_in_order:
        workflow_df = summary[summary["workflow"] == workflow]

        if workflow_df.empty:
            continue

        ax.plot(
            workflow_df["roundIndex"],
            workflow_df["lineCountErrorRatePercent"],
            marker="o",
            label=workflow_label(workflow),
        )

    annotate_injected_error_round(ax, ERROR_ROUND_INDEX, y_top=90, text_y=95)

    ax.set_title("Line-Count Error Rate over Rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Line-count error rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="AI-supported workflow")

    save_figure(
        fig,
        slug,
        "Line-Count Error by Round",
        "Line-count constraint error rate across rounds for AI-supported workflows.",
    )


def plot_constraints(df):
    plot_constraint_fulfillment_by_round_and_workflow(df)
    plot_constraint_failure_context_by_round(df)
    plot_line_count_error_by_round_ai_workflows(df)
