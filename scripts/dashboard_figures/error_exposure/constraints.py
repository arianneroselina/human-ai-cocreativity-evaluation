"""Main-round complete constraint fulfillment by exposure."""

from __future__ import annotations

import pandas as pd

from scripts.config import MAIN_ROUND_INDICES, WORKFLOW_ORDER
from scripts.dashboard_figures.error_exposure.rate_plot import (
    ExposureRatePlotConfig,
    plot_exposure_workflow_rate,
)
from scripts.dashboard_figures.helpers import pass_summary
from scripts.utils import require_columns


def plot_main_constraint_fulfillment_by_exposure(
    main_df: pd.DataFrame,
) -> None:
    """Show constraint-fulfillment rates by exposure, round, and workflow."""
    slug = "107_main_constraint_fulfillment_by_error_exposure_round_and_workflow"
    required = {
        "roundIndex",
        "workflow",
        "errorExposed",
        "passedNumeric",
    }
    if not require_columns(
        main_df,
        required,
        "Main round constraint fulfillment by exposure and workflow",
    ):
        return

    plot_df = main_df[
        ["roundIndex", "workflow", "errorExposed", "passedNumeric"]
    ].copy()
    plot_df["roundIndex"] = pd.to_numeric(plot_df["roundIndex"], errors="coerce")
    plot_df["passedNumeric"] = pd.to_numeric(plot_df["passedNumeric"], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["roundIndex", "workflow", "errorExposed", "passedNumeric"]
    )
    plot_df["roundIndex"] = plot_df["roundIndex"].astype(int)
    plot_df = plot_df.loc[
        plot_df["roundIndex"].isin(MAIN_ROUND_INDICES)
        & plot_df["workflow"].isin(WORKFLOW_ORDER)
        & plot_df["passedNumeric"].isin([0, 1])
    ].copy()
    if plot_df.empty:
        return

    group_columns = ["errorExposed", "roundIndex", "workflow"]
    summary = pass_summary(plot_df, group_columns)
    if summary.empty:
        return

    counts = (
        plot_df.groupby(group_columns, dropna=False)
        .agg(
            passedCount=("passedNumeric", "sum"),
            observedCount=("passedNumeric", "size"),
        )
        .reset_index()
    )
    summary = summary.merge(counts, on=group_columns, how="left")
    if "totalRounds" not in summary.columns:
        summary["totalRounds"] = summary["observedCount"]

    plot_exposure_workflow_rate(
        summary,
        ExposureRatePlotConfig(
            slug=slug,
            rate_column="passRatePercent",
            event_count_column="passedCount",
            total_count_column="totalRounds",
            y_label="Rounds fulfilling all constraints (%)",
            title=(
                "Complete Constraint Fulfillment by Main Round, "
                "Workflow, and Error Exposure"
            ),
            description=(
                "Complete constraint-fulfillment rates for each Main round and "
                "selected workflow, shown separately by actual Main 1 error "
                "exposure. Points show pass-rate estimates and whiskers show 95% "
                "Wilson confidence intervals."
            ),
            figure_height=5.5,
            layout_bottom=0.09,
        ),
    )
