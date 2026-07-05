"""Constraint fulfillment analysis.

Figures
-------
21  Full constraint-pass rate by workflow in main rounds
22  Constraint-score distribution by workflow in main rounds
23  Failure breakdown by constraint type in main rounds
24  Full constraint fulfillment across Main rounds by error-exposure group
25  Line-count failures in injected-error Main Round 1
26  Line-count failure pattern across Main rounds by error-exposure group
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import (
    ERROR_ROUND_INDEX,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    workflow_display_name,
    exposure_display_name,
    phase_data,
    ordered_exposure_groups,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    parse_bool_or_none,
    require_columns,
    save_figure,
    save_table,
)


CONSTRAINT_TYPE_ORDER = [
    "Line count",
    "Required words",
    "Forbidden words",
    "Total word count",
    "Words per line",
    "Special format",
]

CONSTRAINT_TYPE_CATEGORIES = {
    "lines-": "Line count",
    "must-": "Required words",
    "avoid-": "Forbidden words",
    "maxwords-": "Total word count",
    "maxwpl-": "Words per line",
}

FAILURE_PROFILE_ORDER = [
    "All constraints passed",
    "Line-count rule only",
    "One other rule only",
    "Multiple rules failed",
    "Failure details unavailable",
]

FAILURE_PROFILE_COLORS = {
    "All constraints passed": "#009E73",
    "Line-count rule only": "#E69F00",
    "One other rule only": "#56B4E9",
    "Multiple rules failed": "#D55E00",
    "Failure details unavailable": "#999999",
}

# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def _prepare_constraint_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add constraint-analysis variables to prepared participant-round data."""
    prepared = df.copy()

    if prepared.empty:
        return prepared

    if "passed" in prepared.columns:
        prepared["passedNumeric"] = prepared["passed"].apply(
            lambda value: (
                1.0
                if parse_bool_or_none(value) is True
                else 0.0
                if parse_bool_or_none(value) is False
                else np.nan
            )
        )

    if "constraintScore" in prepared.columns:
        prepared["constraintScore"] = pd.to_numeric(
            prepared["constraintScore"],
            errors="coerce",
        ).clip(lower=0, upper=100)

    if "effectiveTimeMinutes" in prepared.columns:
        prepared["effectiveTimeMinutes"] = pd.to_numeric(
            prepared["effectiveTimeMinutes"],
            errors="coerce",
        )
    elif "timeMs" in prepared.columns:
        prepared["effectiveTimeMinutes"] = (
            pd.to_numeric(prepared["timeMs"], errors="coerce") / 60_000
        )

    return prepared


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a 95% Wilson interval in percentage points."""
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0, (centre - margin) * 100), min(100, (centre + margin) * 100)


def _pass_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarise complete constraint-pass rates with Wilson intervals."""
    data = dataframe.dropna(subset=["passedNumeric"]).copy()
    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(group_columns)["passedNumeric"]
        .agg(totalRounds="count", passedRounds="sum")
        .reset_index()
    )
    summary["passedRounds"] = summary["passedRounds"].astype(int)
    summary["passRatePercent"] = summary["passedRounds"] / summary["totalRounds"] * 100

    intervals = summary.apply(
        lambda row: _wilson_interval(
            int(row["passedRounds"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals
    return summary


def _constraint_type(rule_id: str) -> str:
    for prefix, label in CONSTRAINT_TYPE_CATEGORIES.items():
        if rule_id.startswith(prefix):
            return label
    return "Special format"


def _parse_requirement_results(value) -> list[dict]:
    """Safely parse the JSON requirement-results field."""
    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def _line_count_error(value) -> bool | None:
    """Return whether the single line-count rule failed for one round."""
    for item in _parse_requirement_results(value):
        rule_id = str(item.get("id", ""))
        if not rule_id.startswith("lines-"):
            continue

        passed = parse_bool_or_none(item.get("passed"))
        return None if passed is None else not passed

    return None


def _failure_profile(
    passed_numeric: float,
    requirement_results,
) -> str | None:
    """Classify each round into one mutually exclusive failure profile."""
    if pd.isna(passed_numeric):
        return None

    if passed_numeric == 1:
        return "All constraints passed"

    failed_rule_ids = []

    for item in _parse_requirement_results(requirement_results):
        if parse_bool_or_none(item.get("passed")) is False:
            failed_rule_ids.append(str(item.get("id", "")))

    if not failed_rule_ids:
        return "Failure details unavailable"

    line_count_failures = sum(
        rule_id.startswith("lines-") for rule_id in failed_rule_ids
    )

    if len(failed_rule_ids) == 1:
        if line_count_failures == 1:
            return "Line-count rule only"

        return "One other rule only"

    return "Multiple rules failed"


def _explode_requirement_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create one row per individual requirement check."""
    rows = []

    for _, record in dataframe.iterrows():
        for item in _parse_requirement_results(record.get("requirementResults")):
            passed = parse_bool_or_none(item.get("passed"))
            if passed is None:
                continue

            rule_id = str(item.get("id", ""))
            rows.append(
                {
                    "participantId": record.get("participantId"),
                    "roundIndex": record.get("roundIndex"),
                    "workflow": record.get("workflow"),
                    "errorExposed": record.get("errorExposed"),
                    "constraintType": _constraint_type(rule_id),
                    "constraintLabel": item.get("label", rule_id),
                    "passed": passed,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 21: Compare constraint pass rates by workflow in main rounds
# ---------------------------------------------------------------------------


def plot_main_constraint_pass_rate_by_workflow(main_df) -> None:
    """Compare complete constraint-fulfillment rates across workflows.

    Each point is the observed proportion of main rounds in which every
    task constraint was fulfilled. Horizontal whiskers show 95% Wilson
    confidence intervals.
    """
    slug = "21_main_constraint_pass_rate_by_workflow"

    summary = _pass_summary(main_df, ["workflow"])
    if summary.empty:
        return

    summary = (
        summary.set_index("workflow")
        .reindex(WORKFLOW_ORDER)
        .dropna(subset=["totalRounds"])
        .reset_index()
    )

    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    y_positions = np.arange(len(summary))
    values = summary["passRatePercent"].to_numpy(dtype=float)
    lower_ci = summary["lowerCI"].to_numpy(dtype=float)
    upper_ci = summary["upperCI"].to_numpy(dtype=float)

    lower_errors = values - lower_ci
    upper_errors = upper_ci - values

    # Light reference lines make percentage comparisons easier.
    for x_position in [0, 25, 50, 75, 100]:
        ax.axvline(
            x_position,
            color="#e6e6e6",
            linewidth=0.9,
            zorder=0,
        )

    # Confidence intervals first, so points remain clear above them.
    ax.errorbar(
        values,
        y_positions,
        xerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.4,
        capsize=4,
        capthick=1.4,
        zorder=2,
    )

    # One observed pass-rate point per workflow.
    for position, (_, row) in enumerate(summary.iterrows()):
        workflow_color = WORKFLOW_COLORS[row["workflow"]]

        ax.scatter(
            row["passRatePercent"],
            position,
            s=95,
            color=workflow_color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        label = (
            f"{int(row['passedRounds'])}/{int(row['totalRounds'])} "
            f"({row['passRatePercent']:.1f}%)"
        )

        ax.text(
            105,
            position,
            label,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["workflowLabel"], fontsize=10)
    ax.invert_yaxis()

    # Here intermediate values are valid because this axis represents
    # workflow-level pass-rate estimates, not the individual binary outcomes.
    ax.set_xlim(-2, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_xlabel("Rounds fully meeting every constraint", labelpad=10)

    ax.set_title("Complete Constraint Fulfillment Rate by Workflow")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
        pad=10,
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    fig.subplots_adjust(
        left=0.23,
        right=0.96,
        top=0.84,
        bottom=0.18,
    )

    save_figure(
        fig,
        slug,
        "Complete Constraint Fulfillment Rate by Workflow",
        "Points show the observed percentage of main-round poems that fulfilled "
        "every task constraint. Horizontal whiskers show 95% Wilson confidence "
        "intervals. Labels show the number of fully successful rounds out of all "
        "evaluated rounds for each workflow.",
    )


# ---------------------------------------------------------------------------
# 22: Constraint-failure profiles by workflow in main rounds
# ---------------------------------------------------------------------------


def plot_main_constraint_failure_profile_by_workflow(main_df) -> None:
    """Show why Main-round outputs did not fully meet all constraints."""
    slug = "22_main_constraint_failure_profile_by_workflow"

    profile_df = main_df.dropna(subset=["passedNumeric", "workflow"]).copy()

    profile_df["failureProfile"] = [
        _failure_profile(
            row.passedNumeric,
            row.requirementResults,
        )
        for row in profile_df.itertuples(index=False)
    ]

    profile_df = profile_df.dropna(subset=["failureProfile"]).copy()
    if profile_df.empty:
        return

    workflow_order = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow in set(profile_df["workflow"])
    ]

    observed_profiles = [
        profile
        for profile in FAILURE_PROFILE_ORDER
        if profile in set(profile_df["failureProfile"])
    ]

    if not workflow_order or not observed_profiles:
        return

    counts = pd.crosstab(
        profile_df["workflow"],
        profile_df["failureProfile"],
    ).reindex(
        index=workflow_order,
        columns=observed_profiles,
        fill_value=0,
    )

    counts = counts.loc[counts.sum(axis=1).gt(0)]
    if counts.empty:
        return

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    summary = (
        counts.rename_axis(
            index="workflow",
            columns="failureProfile",
        )
        .stack()
        .rename("rounds")
        .reset_index()
    )
    summary["totalRounds"] = summary["workflow"].map(counts.sum(axis=1))
    summary["percentage"] = summary["rounds"] / summary["totalRounds"] * 100
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    positions = np.arange(len(counts))
    left = np.zeros(len(counts), dtype=float)

    for profile in observed_profiles:
        values = percentages[profile].to_numpy(dtype=float)
        round_counts = counts[profile].to_numpy(dtype=int)

        ax.barh(
            positions,
            values,
            left=left,
            color=FAILURE_PROFILE_COLORS[profile],
            edgecolor="white",
            linewidth=0.8,
            label=profile,
        )

        for position, value, count, start in zip(
            positions,
            values,
            round_counts,
            left,
        ):
            if value >= 9:
                ax.text(
                    start + value / 2,
                    position,
                    f"{count}\n{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

        left += values

    for position, total in enumerate(counts.sum(axis=1)):
        ax.text(
            102,
            position,
            f"n={int(total)}",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [workflow_display_name(workflow) for workflow in counts.index],
        fontsize=10,
    )
    ax.invert_yaxis()

    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of Main-round outputs (%)")
    ax.set_title("Constraint Failure Profiles by Workflow in Main Rounds")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )

    apply_standard_axes_style(ax, grid_axis="x")

    fig.subplots_adjust(
        left=0.21,
        right=0.95,
        top=0.86,
        bottom=0.28,
    )

    save_figure(
        fig,
        slug,
        "Constraint Failure Profiles by Workflow in Main Rounds",
        "Each bar represents all Main-round outputs within one selected workflow. "
        "Categories are mutually exclusive. “Line-count rule only” means that "
        "the line-count check was the sole failed requirement; “Multiple rules "
        "failed” can include the line-count rule alongside other failed checks.",
    )


# ---------------------------------------------------------------------------
# 23: Failure breakdown by constraint type in main rounds
# ---------------------------------------------------------------------------


def plot_main_failure_breakdown_by_constraint_type(main_df) -> None:
    """Identify requirement types that produce failures under each workflow."""
    slug = "23_main_failure_breakdown_by_constraint_type"

    exploded = _explode_requirement_results(main_df)
    if exploded.empty:
        return

    summary = (
        exploded.groupby(["workflow", "constraintType"])["passed"]
        .agg(totalChecks="count", passedChecks="sum")
        .reset_index()
    )
    summary["failureRatePercent"] = (
        1 - summary["passedChecks"] / summary["totalChecks"]
    ) * 100
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    observed_types = [
        constraint_type
        for constraint_type in CONSTRAINT_TYPE_ORDER
        if constraint_type in set(summary["constraintType"])
    ]
    workflow_order = [
        workflow for workflow in WORKFLOW_ORDER if workflow in set(summary["workflow"])
    ]

    failure_matrix = summary.pivot(
        index="workflow",
        columns="constraintType",
        values="failureRatePercent",
    ).reindex(index=workflow_order, columns=observed_types)
    check_matrix = summary.pivot(
        index="workflow",
        columns="constraintType",
        values="totalChecks",
    ).reindex(index=workflow_order, columns=observed_types)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.55 * len(observed_types) + 2.8), 4.7))
    masked = np.ma.masked_invalid(failure_matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, vmin=0, vmax=100, cmap="Reds", aspect="auto")

    for row_index, workflow in enumerate(workflow_order):
        for col_index, constraint_type in enumerate(observed_types):
            rate = failure_matrix.iloc[row_index, col_index]
            checks = check_matrix.iloc[row_index, col_index]
            label = "–" if pd.isna(rate) else f"{rate:.0f}%\nn={int(checks)}"
            text_color = "white" if not pd.isna(rate) and rate >= 55 else "black"
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    ax.set_xticks(range(len(observed_types)))
    ax.set_xticklabels(observed_types, rotation=20, ha="right")
    ax.set_yticks(range(len(workflow_order)))
    ax.set_yticklabels([workflow_display_name(workflow) for workflow in workflow_order])
    ax.set_xlabel("Constraint type")
    ax.set_ylabel("Assigned workflow")
    ax.set_title("Constraint Failure Breakdown by Type in Main Rounds")
    fig.colorbar(image, ax=ax, label="Failure rate (%)")

    save_figure(
        fig,
        slug,
        "Constraint Failure Breakdown by Type in Main Rounds",
        "Failure rates by requirement type and assigned workflow. Each cell also "
        "shows the number of individual requirement checks supporting the rate.",
    )


# ---------------------------------------------------------------------------
# 24: Full constraint fulfillment by error exposure in main rounds
# ---------------------------------------------------------------------------


def plot_main_constraint_fulfillment_by_exposure(main_df) -> None:
    """Show descriptive Main-round pass-rate patterns after the error opportunity."""
    slug = "24_main_constraint_fulfillment_by_error_exposure"

    main_df = main_df.dropna(subset=["errorExposed", "passedNumeric"])
    groups = ordered_exposure_groups(main_df)
    if main_df.empty or not groups:
        return

    main_rounds = sorted(main_df["roundIndex"].unique().tolist())
    summary = _pass_summary(
        main_df,
        ["errorExposed", "roundIndex"],
    )
    if summary.empty:
        return

    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    offsets = np.linspace(-0.08, 0.08, len(groups))

    for offset, group in zip(offsets, groups):
        group_summary = (
            summary[summary["errorExposed"].eq(group)]
            .set_index("roundIndex")
            .reindex(main_rounds)
            .dropna(subset=["passRatePercent"])
        )
        if group_summary.empty:
            continue

        x_values = group_summary.index.to_numpy(dtype=float) + offset
        y_values = group_summary["passRatePercent"].to_numpy(dtype=float)
        lower_errors = y_values - group_summary["lowerCI"].to_numpy(dtype=float)
        upper_errors = group_summary["upperCI"].to_numpy(dtype=float) - y_values

        ax.errorbar(
            x_values,
            y_values,
            yerr=np.vstack([lower_errors, upper_errors]),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=exposure_display_name(group),
        )

        for x_value, (_, row) in zip(x_values, group_summary.iterrows()):
            ax.text(
                x_value,
                max(2, row["lowerCI"] - 6),
                f"n={int(row['totalRounds'])}",
                ha="center",
                va="top",
                fontsize=8,
            )

    ax.axvline(
        ERROR_ROUND_INDEX,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.6,
    )
    ax.text(
        ERROR_ROUND_INDEX + 0.05,
        103,
        "Injected-error round",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.set_xticks(main_rounds)
    ax.set_xticklabels([f"Main {index + 1}" for index in range(len(main_rounds))])
    ax.set_ylim(-8, 113)
    ax.set_xlabel("Free-choice main round")
    ax.set_ylabel("Rounds with all constraints fulfilled (%)")
    ax.set_title("Constraint Fulfillment Across Main Rounds by Error Exposure")
    ax.legend(title="Exposure group", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Constraint Fulfillment Across Main Rounds by Error Exposure",
        "Descriptive complete constraint-pass rates across Main rounds, "
        "stratified by actual error exposure. Error bars show 95% Wilson intervals.",
    )


# ---------------------------------------------------------------------------
# 25: Main Round 1 line-count manipulation check
# ---------------------------------------------------------------------------


def plot_injected_error_round_line_count_failures(main_df) -> None:
    """Show line-count failures in the round containing the injected AI error."""
    slug = "25_injected_error_round_line_count_failures"

    error_round_df = main_df[main_df["roundIndex"].eq(ERROR_ROUND_INDEX)].copy()
    error_round_df["lineCountError"] = error_round_df["requirementResults"].apply(
        _line_count_error
    )
    error_round_df = error_round_df.dropna(subset=["lineCountError", "errorExposed"])
    if error_round_df.empty:
        return

    error_round_df["lineCountError"] = error_round_df["lineCountError"].astype(bool)
    summary = (
        error_round_df.groupby(["errorExposed", "workflow"])["lineCountError"]
        .agg(totalRounds="count", failedLineCount="sum")
        .reset_index()
    )
    summary["failedLineCount"] = summary["failedLineCount"].astype(int)
    summary["lineCountFailureRatePercent"] = (
        summary["failedLineCount"] / summary["totalRounds"] * 100
    )

    intervals = summary.apply(
        lambda row: _wilson_interval(
            int(row["failedLineCount"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    save_table(summary, slug, index=False)

    categories = [
        (group, workflow)
        for group in ordered_exposure_groups(error_round_df)
        for workflow in WORKFLOW_ORDER
        if (
            (summary["errorExposed"].eq(group)) & (summary["workflow"].eq(workflow))
        ).any()
    ]
    if not categories:
        return

    plot_summary = pd.DataFrame(
        [
            summary[
                summary["errorExposed"].eq(group) & summary["workflow"].eq(workflow)
            ].iloc[0]
            for group, workflow in categories
        ]
    )

    labels = [
        f"{exposure_display_name(group)}\n{workflow_display_name(workflow)}"
        for group, workflow in categories
    ]
    colors = [WORKFLOW_COLORS[workflow] for _, workflow in categories]
    positions = np.arange(len(plot_summary))
    values = plot_summary["lineCountFailureRatePercent"].to_numpy(dtype=float)
    lower_errors = values - plot_summary["lowerCI"].to_numpy(dtype=float)
    upper_errors = plot_summary["upperCI"].to_numpy(dtype=float) - values

    fig, ax = plt.subplots(figsize=(max(8.6, 1.65 * len(labels)), 5.2))
    ax.bar(
        positions,
        values,
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.1,
        zorder=4,
    )

    for index, (_, row) in enumerate(plot_summary.iterrows()):
        ax.text(
            index,
            min(104, row["upperCI"] + 3),
            (
                f"{int(row['failedLineCount'])}/{int(row['totalRounds'])}\n"
                f"({row['lineCountFailureRatePercent']:.1f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 114)
    ax.set_xlabel("Main Round 1 group and selected workflow")
    ax.set_ylabel("Line-count failure rate (%)")
    ax.set_title("Line-Count Failures in the Injected-Error Round")
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Line-Count Failures in the Injected-Error Round",
        "Line-count constraint failures in Main Round 1, grouped by actual error "
        "exposure and selected workflow. Error bars show 95% Wilson intervals.",
    )


# ---------------------------------------------------------------------------
# 26: Line-count failure rate by error exposure in main rounds
# ---------------------------------------------------------------------------


def plot_main_line_count_error_by_exposure(main_df) -> None:
    """Show descriptive line-count failure patterns across Main rounds."""
    slug = "26_main_line_count_error_by_error_exposure"

    main_df["lineCountError"] = main_df["requirementResults"].apply(_line_count_error)
    main_df = main_df.dropna(subset=["lineCountError", "errorExposed"])
    if main_df.empty:
        return

    main_df["lineCountError"] = main_df["lineCountError"].astype(bool)
    summary = (
        main_df.groupby(["errorExposed", "roundIndex"])["lineCountError"]
        .agg(totalRounds="count", failedLineCount="sum")
        .reset_index()
    )
    summary["failedLineCount"] = summary["failedLineCount"].astype(int)
    summary["lineCountFailureRatePercent"] = (
        summary["failedLineCount"] / summary["totalRounds"] * 100
    )
    intervals = summary.apply(
        lambda row: _wilson_interval(
            int(row["failedLineCount"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    save_table(summary, slug, index=False)

    groups = ordered_exposure_groups(main_df)
    main_rounds = sorted(main_df["roundIndex"].unique().tolist())
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    offsets = np.linspace(-0.08, 0.08, len(groups))

    for offset, group in zip(offsets, groups):
        group_summary = (
            summary[summary["errorExposed"].eq(group)]
            .set_index("roundIndex")
            .reindex(main_rounds)
            .dropna(subset=["lineCountFailureRatePercent"])
        )
        if group_summary.empty:
            continue

        x_values = group_summary.index.to_numpy(dtype=float) + offset
        values = group_summary["lineCountFailureRatePercent"].to_numpy(dtype=float)
        lower_errors = values - group_summary["lowerCI"].to_numpy(dtype=float)
        upper_errors = group_summary["upperCI"].to_numpy(dtype=float) - values

        ax.errorbar(
            x_values,
            values,
            yerr=np.vstack([lower_errors, upper_errors]),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=exposure_display_name(group),
        )

        for x_value, (_, row) in zip(x_values, group_summary.iterrows()):
            ax.text(
                x_value,
                max(1, row["lowerCI"] - 5),
                f"n={int(row['totalRounds'])}",
                ha="center",
                va="top",
                fontsize=8,
            )

    ax.axvline(
        ERROR_ROUND_INDEX,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.6,
    )
    ax.text(
        ERROR_ROUND_INDEX + 0.05,
        103,
        "Injected-error round",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    ax.set_xticks(main_rounds)
    ax.set_xticklabels([f"Main {index + 1}" for index in range(len(main_rounds))])
    ax.set_ylim(-8, 113)
    ax.set_xlabel("Free-choice main round")
    ax.set_ylabel("Line-count failure rate (%)")
    ax.set_title("Line-Count Failure Pattern Across Main Rounds by Error Exposure")
    ax.legend(title="Exposure group", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Line-Count Failure Pattern Across Main Rounds by Error Exposure",
        "Descriptive line-count failure rates across Main rounds, stratified by "
        "actual error exposure. Error bars show 95% Wilson intervals.",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_constraints(df: pd.DataFrame) -> None:
    """Generate controlled, Main-round, and diagnostic constraint figures."""
    prepared = _prepare_constraint_data(df)
    main_df = phase_data(prepared, "main")

    required = {"requirementResults", "errorExposed", "passedNumeric"}
    if main_df.empty or not require_columns(
        main_df,
        required,
        "main-round constraint fulfillment",
    ):
        return

    plot_main_constraint_pass_rate_by_workflow(main_df)
    plot_main_constraint_failure_profile_by_workflow(main_df)
    plot_main_failure_breakdown_by_constraint_type(main_df)
    plot_main_constraint_fulfillment_by_exposure(main_df)
    plot_injected_error_round_line_count_failures(main_df)
    plot_main_line_count_error_by_exposure(main_df)
