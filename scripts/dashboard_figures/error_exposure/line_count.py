"""Main-round line-count failures by exposure."""

from __future__ import annotations

import pandas as pd

from scripts.config import MAIN_ROUND_INDICES, WORKFLOW_ORDER
from scripts.dashboard_figures.error_exposure.rate_plot import (
    ExposureRatePlotConfig,
    plot_exposure_workflow_rate,
)
from scripts.dashboard_figures.helpers import (
    parse_requirement_results,
    wilson_interval,
)
from scripts.utils import parse_bool_or_none, require_columns


def _line_count_error(value: object) -> bool | None:
    """Return whether the single line-count rule failed for one round."""
    for item in parse_requirement_results(value):
        rule_id = str(item.get("id", ""))
        if not rule_id.startswith("lines-"):
            continue

        passed = parse_bool_or_none(item.get("passed"))
        return None if passed is None else not passed

    return None


def plot_main_line_count_error_by_exposure(
    main_df: pd.DataFrame,
) -> None:
    """Show line-count failure rates by exposure, Main round, and workflow."""
    slug = "108_main_line_count_error_by_error_exposure_round_and_workflow"
    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "requirementResults",
    }
    if not require_columns(
        main_df,
        required,
        "Main round line-count failures by exposure and workflow",
    ):
        return

    plot_df = main_df[
        ["roundIndex", "workflow", "errorExposed", "requirementResults"]
    ].copy()
    plot_df["roundIndex"] = pd.to_numeric(plot_df["roundIndex"], errors="coerce")
    plot_df["lineCountError"] = plot_df["requirementResults"].apply(_line_count_error)
    plot_df = plot_df.dropna(
        subset=["roundIndex", "workflow", "errorExposed", "lineCountError"]
    )
    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)
    plot_df["lineCountError"] = plot_df["lineCountError"].astype(bool)
    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
    ].copy()
    if plot_df.empty:
        return

    group_columns = ["errorExposed", "roundIndex", "workflow"]
    summary = (
        plot_df.groupby(group_columns, dropna=False)["lineCountError"]
        .agg(totalRounds="count", failedLineCount="sum")
        .reset_index()
    )
    summary["totalRounds"] = pd.to_numeric(
        summary["totalRounds"], errors="coerce"
    ).astype(int)
    summary["failedLineCount"] = pd.to_numeric(
        summary["failedLineCount"], errors="coerce"
    ).astype(int)
    summary["lineCountFailureRatePercent"] = (
        summary["failedLineCount"] / summary["totalRounds"] * 100
    )
    intervals = summary.apply(
        lambda row: wilson_interval(
            int(row["failedLineCount"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals

    plot_exposure_workflow_rate(
        summary,
        ExposureRatePlotConfig(
            slug=slug,
            rate_column="lineCountFailureRatePercent",
            event_count_column="failedLineCount",
            total_count_column="totalRounds",
            y_label="Rounds with a line-count failure (%)",
            title=(
                "Line-Count Failure Rate by Main Round, Workflow, and Error Exposure"
            ),
            description=(
                "Line-count failure rates for each Main round and selected "
                "workflow, shown separately by actual Main 1 error exposure. "
                "Points show failure-rate estimates, whiskers show 95% Wilson "
                "confidence intervals, and labels report failed rounds divided "
                "by observed rounds."
            ),
        ),
    )
