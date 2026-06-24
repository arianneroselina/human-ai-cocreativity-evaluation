"""Constraint fulfillment analysis.

Figures
-------
21  Full constraint-pass rate by workflow — controlled Practice rounds
22  Constraint-score distribution by workflow — controlled Practice rounds
23  Failure breakdown by constraint type — controlled Practice rounds
24  Full constraint fulfillment across Main rounds by error-exposure group
25  Line-count failures in injected-error Main Round 1
26  Line-count failure pattern across Main rounds by error-exposure group
27  Constraint fulfillment versus external text quality

Optional
--------
28  Recorded task time by line-count outcome

Notes
-----
- ``passed`` means every requirement for a round was fulfilled.
- ``constraintScore`` is stored on a 0–100 scale.
- Error-exposure figures are descriptive: exposure followed participants'
  voluntary Main Round 1 workflow selection.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.config import (
    ERROR_ROUND_INDEX,
    EXPOSURE_LABELS,
    MAIN_ROUND_INDICES,
    PRACTICE_ROUND_INDICES,
    QUALITY_PRIMARY_METRIC,
    QUALITY_SCALE_MAX,
    QUALITY_SCALE_MIN,
    WORKFLOW_COLORS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    drop_duplicate_participant_rounds,
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


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


def _workflow_label(workflow: str) -> str:
    return WORKFLOW_LABELS.get(workflow, str(workflow))


def _exposure_label(exposure_group: str) -> str:
    return EXPOSURE_LABELS.get(
        str(exposure_group),
        str(exposure_group).replace("_", " ").title(),
    )


def _ordered_exposure_groups(dataframe: pd.DataFrame) -> list[str]:
    """Return canonical exposure groups followed by any unexpected values."""
    if "errorExposureGroup" not in dataframe.columns:
        return []

    available = set(dataframe["errorExposureGroup"].dropna().astype(str))
    ordered = [
        group
        for group in ["error_exposed", "not_exposed"]
        if group in available
    ]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _prepare_constraint_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one observation per participant-round and normalize key variables."""
    required = {"participantId", "roundIndex", "workflow"}
    if df.empty or not require_columns(df, required, "constraint data"):
        return pd.DataFrame()

    prepared = drop_duplicate_participant_rounds(df.copy())
    prepared["roundIndex"] = pd.to_numeric(
        prepared["roundIndex"],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=["participantId", "roundIndex", "workflow"])
    prepared["roundIndex"] = prepared["roundIndex"].astype(int)
    prepared = prepared[prepared["workflow"].isin(WORKFLOW_ORDER)].copy()

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


def _phase_data(
        prepared: pd.DataFrame,
        phase: str,
) -> pd.DataFrame:
    """Return Practice or Main observations with a round-index fallback."""
    if prepared.empty:
        return prepared.copy()

    if phase == "practice":
        if "isPracticeRound" in prepared.columns:
            return prepared[prepared["isPracticeRound"].eq(True)].copy()
        return prepared[
            prepared["roundIndex"].isin(PRACTICE_ROUND_INDICES)
        ].copy()

    if phase == "main":
        if "isMainRound" in prepared.columns:
            return prepared[prepared["isMainRound"].eq(True)].copy()
        return prepared[
            prepared["roundIndex"].isin(MAIN_ROUND_INDICES)
        ].copy()

    raise ValueError(f"Unknown phase: {phase}")


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
            * np.sqrt(
        proportion * (1 - proportion) / total
        + z**2 / (4 * total**2)
    )
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
    summary["passRatePercent"] = (
            summary["passedRounds"] / summary["totalRounds"] * 100
    )

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
                    "errorExposureGroup": record.get("errorExposureGroup"),
                    "constraintType": _constraint_type(rule_id),
                    "constraintLabel": item.get("label", rule_id),
                    "passed": passed,
                }
            )

    return pd.DataFrame(rows)


def _draw_raw_points(
        ax,
        x_positions: Iterable[float],
        values: Iterable[float],
        color: str,
        seed: int,
        alpha: float = 0.45,
) -> None:
    """Draw deterministic horizontally jittered points."""
    values = np.asarray(list(values), dtype=float)
    positions = np.asarray(list(x_positions), dtype=float)
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 0.055, size=len(values))
    ax.scatter(
        positions + jitter,
        values,
        s=22,
        color=color,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        )


# ---------------------------------------------------------------------------
# 21 – Controlled Practice comparison: complete pass rate
# ---------------------------------------------------------------------------


def plot_practice_constraint_pass_rate_by_workflow(df: pd.DataFrame) -> None:
    """Compare complete requirement fulfillment under assigned workflows."""
    slug = "21_practice_constraint_pass_rate_by_workflow"
    prepared = _prepare_constraint_data(df)
    practice_df = _phase_data(prepared, "practice")

    if practice_df.empty or "passedNumeric" not in practice_df.columns:
        return

    summary = _pass_summary(practice_df, ["workflow"])
    if summary.empty:
        return

    summary = (
        summary.set_index("workflow")
        .reindex(WORKFLOW_ORDER)
        .dropna(subset=["totalRounds"])
        .reset_index()
    )
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    positions = np.arange(len(summary))
    values = summary["passRatePercent"].to_numpy(dtype=float)
    lower_errors = values - summary["lowerCI"].to_numpy(dtype=float)
    upper_errors = summary["upperCI"].to_numpy(dtype=float) - values

    bars = ax.bar(
        positions,
        values,
        color=[WORKFLOW_COLORS[workflow] for workflow in summary["workflow"]],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )
    ax.errorbar(
        positions,
        values,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
        zorder=4,
    )

    for index, row in summary.iterrows():
        workflow_data = practice_df.loc[
                            practice_df["workflow"].eq(row["workflow"])
                            & practice_df["passedNumeric"].notna(),
                            "passedNumeric",
                        ] * 100
        _draw_raw_points(
            ax,
            [index] * len(workflow_data),
            workflow_data,
            WORKFLOW_COLORS[row["workflow"]],
            seed=index + 11,
            )
        ax.text(
            index,
            min(104, row["upperCI"] + 3),
            (
                f"{int(row['passedRounds'])}/{int(row['totalRounds'])}\n"
                f"({row['passRatePercent']:.1f}%)"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(summary["workflowLabel"], rotation=12, ha="right")
    ax.set_ylim(-4, 114)
    ax.set_xlabel("Assigned workflow")
    ax.set_ylabel("Rounds with all constraints fulfilled (%)")
    ax.set_title("Complete Constraint Fulfillment by Workflow — Practice Rounds")
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Complete Constraint Fulfillment by Workflow — Practice Rounds",
        "Controlled comparison of the percentage of rounds in which every task "
        "constraint was fulfilled. Error bars show 95% Wilson confidence intervals.",
    )


# ---------------------------------------------------------------------------
# 22 – Controlled Practice comparison: score distribution
# ---------------------------------------------------------------------------


def plot_practice_constraint_score_distribution(df: pd.DataFrame) -> None:
    """Show how close each workflow came to full fulfillment in Practice rounds."""
    slug = "22_practice_constraint_score_distribution_by_workflow"
    prepared = _prepare_constraint_data(df)
    practice_df = _phase_data(prepared, "practice").dropna(
        subset=["constraintScore"]
    )

    if practice_df.empty:
        return

    workflow_order = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow in set(practice_df["workflow"])
    ]
    if not workflow_order:
        return

    summary = (
        practice_df.groupby("workflow")["constraintScore"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
            count="count",
        )
        .reindex(workflow_order)
        .reset_index()
    )
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    save_table(summary, slug, index=False)

    figure_data = [
        practice_df.loc[
            practice_df["workflow"].eq(workflow),
            "constraintScore",
        ].to_numpy(dtype=float)
        for workflow in workflow_order
    ]

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    boxplot = ax.boxplot(
        figure_data,
        tick_labels=[_workflow_label(workflow) for workflow in workflow_order],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "white",
            "markersize": 6,
        },
    )

    for patch, workflow in zip(boxplot["boxes"], workflow_order):
        patch.set_facecolor(WORKFLOW_COLORS[workflow])
        patch.set_alpha(0.65)

    for index, (workflow, values) in enumerate(
            zip(workflow_order, figure_data),
            start=1,
    ):
        _draw_raw_points(
            ax,
            [index] * len(values),
            values,
            WORKFLOW_COLORS[workflow],
            seed=index + 31,
            )
        ax.text(
            index,
            2.5,
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(
        100,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.65,
    )
    ax.text(
        len(workflow_order) + 0.25,
        100,
        "All constraints passed",
        va="center",
        fontsize=8,
        )
    ax.set_ylim(0, 111)
    ax.set_xlabel("Assigned workflow")
    ax.set_ylabel("Constraint score (0–100)")
    ax.set_title("Constraint Score Distribution by Workflow — Practice Rounds")
    ax.tick_params(axis="x", rotation=12)
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Constraint Score Distribution by Workflow — Practice Rounds",
        "Distribution of partial constraint fulfillment under assigned workflows. "
        "A score of 100 indicates that all constraints were fulfilled.",
    )


# ---------------------------------------------------------------------------
# 23 – Controlled Practice comparison: failure type breakdown
# ---------------------------------------------------------------------------


def plot_practice_failure_breakdown_by_constraint_type(df: pd.DataFrame) -> None:
    """Identify requirement types that produce failures under each workflow."""
    slug = "23_practice_failure_breakdown_by_constraint_type"
    prepared = _prepare_constraint_data(df)
    practice_df = _phase_data(prepared, "practice")

    if practice_df.empty or "requirementResults" not in practice_df.columns:
        return

    exploded = _explode_requirement_results(practice_df)
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
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    save_table(summary, slug, index=False)

    observed_types = [
        constraint_type
        for constraint_type in CONSTRAINT_TYPE_ORDER
        if constraint_type in set(summary["constraintType"])
    ]
    workflow_order = [
        workflow
        for workflow in WORKFLOW_ORDER
        if workflow in set(summary["workflow"])
    ]

    failure_matrix = (
        summary.pivot(
            index="workflow",
            columns="constraintType",
            values="failureRatePercent",
        )
        .reindex(index=workflow_order, columns=observed_types)
    )
    check_matrix = (
        summary.pivot(
            index="workflow",
            columns="constraintType",
            values="totalChecks",
        )
        .reindex(index=workflow_order, columns=observed_types)
    )

    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.55 * len(observed_types) + 2.8), 4.7)
    )
    masked = np.ma.masked_invalid(failure_matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, vmin=0, vmax=100, cmap="Reds", aspect="auto")

    for row_index, workflow in enumerate(workflow_order):
        for col_index, constraint_type in enumerate(observed_types):
            rate = failure_matrix.iloc[row_index, col_index]
            checks = check_matrix.iloc[row_index, col_index]
            label = (
                "–"
                if pd.isna(rate)
                else f"{rate:.0f}%\nn={int(checks)}"
            )
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
    ax.set_yticklabels([_workflow_label(workflow) for workflow in workflow_order])
    ax.set_xlabel("Constraint type")
    ax.set_ylabel("Assigned workflow")
    ax.set_title("Constraint Failure Breakdown by Type — Practice Rounds")
    fig.colorbar(image, ax=ax, label="Failure rate (%)")

    save_figure(
        fig,
        slug,
        "Constraint Failure Breakdown by Type — Practice Rounds",
        "Failure rates by requirement type and assigned workflow. Each cell also "
        "shows the number of individual requirement checks supporting the rate.",
    )


# ---------------------------------------------------------------------------
# 24 – Main rounds: full pass rate by exposure group
# ---------------------------------------------------------------------------


def plot_main_constraint_fulfillment_by_exposure(df: pd.DataFrame) -> None:
    """Show descriptive Main-round pass-rate patterns after the error opportunity."""
    slug = "24_main_constraint_fulfillment_by_error_exposure"
    prepared = _prepare_constraint_data(df)
    main_df = _phase_data(prepared, "main")

    required = {"errorExposureGroup", "passedNumeric"}
    if main_df.empty or not require_columns(
            main_df,
            required,
            "main-round constraint fulfillment",
    ):
        return

    main_df = main_df.dropna(subset=["errorExposureGroup", "passedNumeric"])
    groups = _ordered_exposure_groups(main_df)
    if main_df.empty or not groups:
        return

    main_rounds = sorted(main_df["roundIndex"].unique().tolist())
    summary = _pass_summary(
        main_df,
        ["errorExposureGroup", "roundIndex"],
    )
    if summary.empty:
        return

    summary["exposureLabel"] = summary["errorExposureGroup"].map(_exposure_label)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    offsets = np.linspace(-0.08, 0.08, len(groups))

    for offset, group in zip(offsets, groups):
        group_summary = (
            summary[summary["errorExposureGroup"].eq(group)]
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
            label=_exposure_label(group),
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
    ax.set_xticklabels(
        [f"Main {index + 1}" for index in range(len(main_rounds))]
    )
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
        "Descriptive complete constraint-pass rates across free-choice Main rounds, "
        "stratified by actual error exposure. Error bars show 95% Wilson intervals.",
    )


# ---------------------------------------------------------------------------
# 25 – Main Round 1 line-count manipulation check
# ---------------------------------------------------------------------------


def plot_injected_error_round_line_count_failures(df: pd.DataFrame) -> None:
    """Show line-count failures in the round containing the injected AI error."""
    slug = "25_injected_error_round_line_count_failures"
    prepared = _prepare_constraint_data(df)

    required = {"requirementResults", "errorExposureGroup"}
    if prepared.empty or not require_columns(
            prepared,
            required,
            "injected-error line-count analysis",
    ):
        return

    error_round_df = prepared[
        prepared["roundIndex"].eq(ERROR_ROUND_INDEX)
    ].copy()
    error_round_df["lineCountError"] = error_round_df[
        "requirementResults"
    ].apply(_line_count_error)
    error_round_df = error_round_df.dropna(
        subset=["lineCountError", "errorExposureGroup"]
    )
    if error_round_df.empty:
        return

    error_round_df["lineCountError"] = (
        error_round_df["lineCountError"].astype(bool)
    )
    summary = (
        error_round_df.groupby(["errorExposureGroup", "workflow"])[
            "lineCountError"
        ]
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
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    summary["exposureLabel"] = summary["errorExposureGroup"].map(_exposure_label)
    save_table(summary, slug, index=False)

    categories = [
        (group, workflow)
        for group in _ordered_exposure_groups(error_round_df)
        for workflow in WORKFLOW_ORDER
        if (
                (summary["errorExposureGroup"].eq(group))
                & (summary["workflow"].eq(workflow))
        ).any()
    ]
    if not categories:
        return

    plot_summary = pd.DataFrame(
        [
            summary[
                summary["errorExposureGroup"].eq(group)
                & summary["workflow"].eq(workflow)
                ].iloc[0]
            for group, workflow in categories
        ]
    )

    labels = [
        f"{_exposure_label(group)}\n{_workflow_label(workflow)}"
        for group, workflow in categories
    ]
    colors = [
        WORKFLOW_COLORS[workflow]
        for _, workflow in categories
    ]
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
# 26 – Main rounds: line-count error pattern and recovery
# ---------------------------------------------------------------------------


def plot_main_line_count_error_by_exposure(df: pd.DataFrame) -> None:
    """Show descriptive line-count failure patterns across Main rounds."""
    slug = "26_main_line_count_error_by_error_exposure"
    prepared = _prepare_constraint_data(df)

    required = {"requirementResults", "errorExposureGroup"}
    if prepared.empty or not require_columns(
            prepared,
            required,
            "main-round line-count analysis",
    ):
        return

    main_df = _phase_data(prepared, "main").copy()
    main_df["lineCountError"] = main_df["requirementResults"].apply(
        _line_count_error
    )
    main_df = main_df.dropna(
        subset=["lineCountError", "errorExposureGroup"]
    )
    if main_df.empty:
        return

    main_df["lineCountError"] = main_df["lineCountError"].astype(bool)
    summary = (
        main_df.groupby(["errorExposureGroup", "roundIndex"])["lineCountError"]
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
    summary["exposureLabel"] = summary["errorExposureGroup"].map(_exposure_label)
    save_table(summary, slug, index=False)

    groups = _ordered_exposure_groups(main_df)
    main_rounds = sorted(main_df["roundIndex"].unique().tolist())
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    offsets = np.linspace(-0.08, 0.08, len(groups))

    for offset, group in zip(offsets, groups):
        group_summary = (
            summary[summary["errorExposureGroup"].eq(group)]
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
            label=_exposure_label(group),
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
    ax.set_xticklabels(
        [f"Main {index + 1}" for index in range(len(main_rounds))]
    )
    ax.set_ylim(-8, 113)
    ax.set_xlabel("Free-choice main round")
    ax.set_ylabel("Line-count failure rate (%)")
    ax.set_title("Line-Count Failure Pattern Across Main Rounds")
    ax.legend(title="Exposure group", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Line-Count Failure Pattern Across Main Rounds",
        "Descriptive line-count failure rates across Main rounds, stratified by "
        "actual error exposure. Error bars show 95% Wilson intervals.",
    )


# ---------------------------------------------------------------------------
# 27 – Constraint fulfillment and external quality
# ---------------------------------------------------------------------------


def plot_constraint_score_vs_quality(df: pd.DataFrame) -> None:
    """Inspect the descriptive relationship between requirement fulfillment and quality."""
    slug = "27_constraint_score_vs_external_quality"
    prepared = _prepare_constraint_data(df)

    required = {"constraintScore", QUALITY_PRIMARY_METRIC, "workflow"}
    if prepared.empty or not require_columns(
            prepared,
            required,
            "constraint score versus external quality",
    ):
        return

    plot_df = prepared.copy()
    plot_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        plot_df[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )
    plot_df = plot_df.dropna(
        subset=["constraintScore", QUALITY_PRIMARY_METRIC, "workflow"]
    )

    if plot_df.empty:
        return

    rho = plot_df["constraintScore"].corr(
        plot_df[QUALITY_PRIMARY_METRIC],
        method="spearman",
    )
    summary = (
        plot_df.groupby("workflow")
        .agg(
            rounds=("participantId", "count"),
            meanConstraintScore=("constraintScore", "mean"),
            meanQuality=(QUALITY_PRIMARY_METRIC, "mean"),
        )
        .reindex(WORKFLOW_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    summary["overallSpearmanRho"] = rho
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.3))

    for index, workflow in enumerate(WORKFLOW_ORDER):
        workflow_df = plot_df[plot_df["workflow"].eq(workflow)]
        if workflow_df.empty:
            continue

        rng = np.random.default_rng(index + 71)
        jitter = rng.normal(0, 0.7, size=len(workflow_df))
        ax.scatter(
            workflow_df["constraintScore"] + jitter,
            workflow_df[QUALITY_PRIMARY_METRIC],
            s=34,
            alpha=0.6,
            color=WORKFLOW_COLORS[workflow],
            edgecolor="white",
            linewidth=0.45,
            label=_workflow_label(workflow),
            )

    ax.axvline(100, linestyle="--", linewidth=1, color="black", alpha=0.6)
    ax.set_xlim(-3, 103)
    ax.set_ylim(QUALITY_SCALE_MIN - 0.15, QUALITY_SCALE_MAX + 0.15)
    ax.set_xlabel("Constraint score (0–100)")
    ax.set_ylabel("External quality composite (1–5)")
    ax.set_title("Constraint Fulfillment and External Text Quality")
    ax.legend(title="Workflow", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="both")

    if pd.notna(rho):
        ax.text(
            0.02,
            0.98,
            f"Overall Spearman ρ = {rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.9,
            },
        )

    save_figure(
        fig,
        slug,
        "Constraint Fulfillment and External Text Quality",
        "Descriptive relationship between constraint fulfillment and independently "
        "rated text quality. The displayed Spearman correlation pools all rounds.",
    )


# ---------------------------------------------------------------------------
# 28 – Optional recorded-time diagnostic
# ---------------------------------------------------------------------------


def plot_recorded_time_by_line_count_outcome(df: pd.DataFrame) -> None:
    """Compare recorded task time for rounds with and without line-count failures.

    This is descriptive only. It does not infer that less time caused a failure,
    especially because pauses may increase recorded task time.
    """
    slug = "28_recorded_time_by_line_count_outcome"
    prepared = _prepare_constraint_data(df)

    required = {"requirementResults", "effectiveTimeMinutes", "workflow"}
    if prepared.empty or not require_columns(
            prepared,
            required,
            "recorded time by line-count outcome",
    ):
        return

    plot_df = prepared.copy()
    plot_df["lineCountError"] = plot_df["requirementResults"].apply(
        _line_count_error
    )
    plot_df = plot_df.dropna(
        subset=["lineCountError", "effectiveTimeMinutes"]
    )
    if plot_df.empty:
        return

    plot_df["lineCountOutcome"] = np.where(
        plot_df["lineCountError"].astype(bool),
        "Line-count failure",
        "No line-count failure",
    )

    summary = (
        plot_df.groupby(["workflow", "lineCountOutcome"])[
            "effectiveTimeMinutes"
        ]
        .agg(
            mean="mean",
            median="median",
            std="std",
            count="count",
        )
        .reset_index()
    )
    summary["workflowLabel"] = summary["workflow"].map(_workflow_label)
    save_table(summary, slug, index=False)

    outcome_order = ["No line-count failure", "Line-count failure"]
    rows = [
        (
            workflow,
            outcome,
            plot_df.loc[
                plot_df["workflow"].eq(workflow)
                & plot_df["lineCountOutcome"].eq(outcome),
                "effectiveTimeMinutes",
            ].to_numpy(dtype=float),
        )
        for workflow in WORKFLOW_ORDER
        for outcome in outcome_order
        if not plot_df.loc[
            plot_df["workflow"].eq(workflow)
            & plot_df["lineCountOutcome"].eq(outcome),
            "effectiveTimeMinutes",
        ].empty
    ]

    if not rows:
        return

    labels = [
        f"{_workflow_label(workflow)}\n{outcome.replace('Line-count ', '')}"
        for workflow, outcome, _ in rows
    ]
    values = [values for _, _, values in rows]
    colors = [
        WORKFLOW_COLORS[workflow]
        for workflow, _, _ in rows
    ]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.35 * len(labels)), 5.2))
    boxplot = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.60)

    for index, (_, _, group_values) in enumerate(rows, start=1):
        _draw_raw_points(
            ax,
            [index] * len(group_values),
            group_values,
            colors[index - 1],
            seed=index + 91,
            )
        ax.text(
            index,
            0.02,
            f"n={len(group_values)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xlabel("Workflow and line-count outcome")
    ax.set_ylabel("Recorded task time (minutes)")
    ax.set_title("Recorded Task Time by Line-Count Outcome")
    ax.tick_params(axis="x", rotation=18)
    apply_standard_axes_style(ax, grid_axis="y")

    save_figure(
        fig,
        slug,
        "Recorded Task Time by Line-Count Outcome",
        "Descriptive recorded task time for rounds with and without line-count "
        "failures. Pauses may contribute to recorded time; this figure does not "
        "support a causal interpretation.",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_constraints(df: pd.DataFrame) -> None:
    """Generate controlled, Main-round, and diagnostic constraint figures."""
    plot_practice_constraint_pass_rate_by_workflow(df)
    plot_practice_constraint_score_distribution(df)
    plot_practice_failure_breakdown_by_constraint_type(df)
    plot_main_constraint_fulfillment_by_exposure(df)
    plot_injected_error_round_line_count_failures(df)
    plot_main_line_count_error_by_exposure(df)
    plot_constraint_score_vs_quality(df)
    plot_recorded_time_by_line_count_outcome(df)
