"""Participant subjective experience analysis.

Figures
-------
31  Satisfaction by practice round position and workflow
32  AI interaction ratings by practice round position and AI-supported workflow
33  Raw NASA-TLX workload by workflow in practice rounds
34  Participant satisfaction versus external text quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    QUALITY_PRIMARY_METRIC,
    QUALITY_SCALE_MAX,
    QUALITY_SCALE_MIN,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
    SATISFACTION_COLUMN,
    AI_EXPERIENCE_METRICS,
    TLX_METRICS,
    PHASES,
)
from scripts.dashboard_figures.helpers import (
    phase_data,
    workflow_display_name,
)
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# Shared preparation and summaries
# ---------------------------------------------------------------------------


def _prepare_experience_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert experience-analysis variables to numeric values."""
    prepared = df.copy()

    numeric_columns = [
        SATISFACTION_COLUMN,
        QUALITY_PRIMARY_METRIC,
        *AI_EXPERIENCE_METRICS,
        *TLX_METRICS,
    ]

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    return prepared


def _available_rounds(dataframe: pd.DataFrame) -> list[int]:
    """Return observed study rounds in ascending order."""
    return sorted(dataframe["roundIndex"].dropna().astype(int).unique().tolist())


def _mean_ci_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    """Calculate descriptive means with approximate 95% confidence intervals."""
    rows = []

    for group_values, group_df in dataframe.groupby(
        group_columns,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_values_dict = dict(zip(group_columns, group_values))

        for metric in metric_columns:
            values = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            count = len(values)
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if count > 1 else np.nan
            standard_error = (
                standard_deviation / np.sqrt(count) if count > 1 else np.nan
            )
            margin = 1.96 * standard_error if np.isfinite(standard_error) else np.nan

            rows.append(
                {
                    **group_values_dict,
                    "metric": metric,
                    "mean": mean,
                    "standardDeviation": standard_deviation,
                    "count": count,
                    "lowerCI": (mean - margin if np.isfinite(margin) else np.nan),
                    "upperCI": (mean + margin if np.isfinite(margin) else np.nan),
                }
            )

    return pd.DataFrame(rows)


def _plot_round_series(
    ax,
    summary: pd.DataFrame,
    workflow: str,
    rounds: list[int],
    metric: str,
    point_offset: float = 0.0,
) -> None:
    """Plot one workflow series across practice round positions."""
    workflow_summary = (
        summary[summary["workflow"].eq(workflow) & summary["metric"].eq(metric)]
        .set_index("roundIndex")
        .reindex(rounds)
    )

    if workflow_summary["mean"].dropna().empty:
        return

    x_values = np.asarray(rounds, dtype=float) + point_offset
    means = workflow_summary["mean"].to_numpy(dtype=float)

    # NaNs intentionally preserve gaps rather than connecting absent cells.
    ax.plot(
        x_values,
        means,
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        color=WORKFLOW_COLORS[workflow],
        label=workflow_display_name(workflow),
        zorder=3,
    )

    valid = workflow_summary["mean"].notna().to_numpy()
    if not valid.any():
        return

    lower_errors = np.full(len(rounds), np.nan)
    upper_errors = np.full(len(rounds), np.nan)
    lower_errors[valid] = means[valid] - workflow_summary.loc[
        workflow_summary["mean"].notna(),
        "lowerCI",
    ].to_numpy(dtype=float)
    upper_errors[valid] = (
        workflow_summary.loc[
            workflow_summary["mean"].notna(),
            "upperCI",
        ].to_numpy(dtype=float)
        - means[valid]
    )

    ax.errorbar(
        x_values[valid],
        means[valid],
        yerr=np.vstack([lower_errors[valid], upper_errors[valid]]),
        fmt="none",
        ecolor=WORKFLOW_COLORS[workflow],
        capsize=3,
        linewidth=1.0,
        alpha=0.9,
        zorder=2,
    )


# ---------------------------------------------------------------------------
# 31: Satisfaction by practice round position and workflow
# ---------------------------------------------------------------------------


def plot_satisfaction_by_practice_round_and_workflow(
    practice_df: pd.DataFrame,
) -> None:
    """Show satisfaction across randomized practice round positions."""
    slug = "31_satisfaction_by_practice_round_and_workflow"

    required_columns = {SATISFACTION_COLUMN, "roundIndex", "workflow"}
    if not require_columns(
        practice_df,
        required_columns,
        "practice round satisfaction by workflow",
    ):
        return

    plot_df = practice_df.dropna(
        subset=[SATISFACTION_COLUMN, "roundIndex", "workflow"]
    ).copy()
    if plot_df.empty:
        return

    rounds = _available_rounds(plot_df)
    summary = _mean_ci_summary(
        plot_df,
        group_columns=["roundIndex", "workflow"],
        metric_columns=[SATISFACTION_COLUMN],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.6, 5.3))

    for workflow in WORKFLOW_ORDER:
        _plot_round_series(
            ax,
            summary,
            workflow,
            rounds,
            SATISFACTION_COLUMN,
        )

    ax.set_title("Participant Satisfaction by Practice Round and Workflow")
    ax.set_xlabel("Practice round")
    ax.set_ylabel("Satisfaction rating (1-5)")
    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Practice {index + 1}" for index in range(len(rounds))])
    ax.set_ylim(0.7, 5.42)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(
        title="Assigned workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    apply_standard_axes_style(ax, grid_axis="y")
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    save_figure(
        fig,
        slug,
        "Participant Satisfaction by Practice Round and Workflow",
        (
            "Mean satisfaction across the randomized practice phase, grouped by "
            "practice round and assigned workflow. Error bars show "
            "approximate 95% confidence intervals; labels show observation counts."
        ),
    )


# ---------------------------------------------------------------------------
# 32: AI interaction ratings by practice round position and workflow
# ---------------------------------------------------------------------------


def plot_ai_experience_by_practice_round_and_workflow(
    practice_df: pd.DataFrame,
) -> None:
    """Show AI interaction ratings across randomized practice rounds."""
    slug = "32_ai_experience_by_practice_round_and_workflow"

    available_metrics = [
        metric
        for metric in AI_EXPERIENCE_METRICS
        if metric in practice_df.columns and practice_df[metric].notna().any()
    ]
    if not available_metrics:
        return

    ai_workflows = [workflow for workflow in WORKFLOW_ORDER if workflow != "human"]
    plot_df = practice_df.loc[practice_df["workflow"].isin(ai_workflows)].copy()
    if plot_df.empty:
        return

    rounds = _available_rounds(plot_df)
    summary = _mean_ci_summary(
        plot_df,
        group_columns=["roundIndex", "workflow"],
        metric_columns=available_metrics,
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["metricLabel"] = summary["metric"].map(AI_EXPERIENCE_METRICS)
    save_table(summary, slug, index=False)

    metric_count = len(available_metrics)
    n_columns = 2
    n_rows = int(np.ceil(metric_count / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12.4, 4.45 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, metric in zip(axes_flat, available_metrics):
        for workflow in ai_workflows:
            _plot_round_series(
                axis,
                summary,
                workflow,
                rounds,
                metric,
            )

        axis.set_title(AI_EXPERIENCE_METRICS[metric])
        axis.set_xticks(rounds)
        axis.set_xticklabels([f"Practice {index + 1}" for index in range(len(rounds))])
        axis.set_ylim(0.7, 5.42)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Practice round")
        axis.set_ylabel("Participant rating (1-5)")
        apply_standard_axes_style(axis, grid_axis="y")

    for axis in axes_flat[len(available_metrics) :]:
        axis.set_visible(False)

    legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        title="Assigned AI-supported workflow",
        bbox_to_anchor=(0.99, 0.5),
        loc="center left",
    )
    fig.suptitle(
        "AI Interaction Ratings by Practice Round and Workflow",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 0.82, 0.97))

    save_figure(
        fig,
        slug,
        "AI Interaction Ratings by Practice Round and Workflow",
        (
            "Mean ratings for AI understanding, collaboration quality, creativity "
            "support, and overall AI performance across the randomized practice "
            "phase. Error bars show approximate 95% confidence intervals; labels "
            "show observation counts."
        ),
    )


# ---------------------------------------------------------------------------
# 33: Raw NASA-TLX workload by workflow in practice rounds
# ---------------------------------------------------------------------------


def plot_tlx_score_by_workflow_in_practice_rounds(
    practice_df: pd.DataFrame,
) -> None:
    """Compare Raw NASA-TLX workload across practice workflows.

    Raw NASA-TLX is the equally weighted mean of the six NASA-TLX subscales
    for each participant-round. Scores remain on the original 0-20 scale.
    """
    slug = "33_tlx_score_by_workflow_in_practice_rounds"

    required_columns = {"workflow", "rawNasaTlxScore"}
    if not require_columns(
        practice_df,
        required_columns,
        "Raw NASA-TLX score by practice workflow",
    ):
        return

    plot_df = practice_df.dropna(subset=["workflow", "rawNasaTlxScore"]).copy()
    if plot_df.empty:
        return

    summary = _mean_ci_summary(
        plot_df,
        group_columns=["workflow"],
        metric_columns=["rawNasaTlxScore"],
    )
    if summary.empty:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    workflow_order = [
        workflow for workflow in WORKFLOW_ORDER if workflow in set(summary["workflow"])
    ]
    plot_summary = (
        summary.set_index("workflow")
        .reindex(workflow_order)
        .dropna(subset=["mean"])
        .reset_index()
    )
    if plot_summary.empty:
        return

    positions = np.arange(len(plot_summary))
    values = plot_summary["mean"].to_numpy(dtype=float)
    lower_ci = plot_summary["lowerCI"].to_numpy(dtype=float)
    upper_ci = plot_summary["upperCI"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.7, 4.8))

    ax.errorbar(
        values,
        positions,
        xerr=np.vstack([values - lower_ci, upper_ci - values]),
        fmt="none",
        ecolor="#303030",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        zorder=2,
    )

    for position, (_, row) in zip(positions, plot_summary.iterrows()):
        workflow = row["workflow"]

        ax.scatter(
            row["mean"],
            position,
            s=95,
            color=WORKFLOW_COLORS[workflow],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{row['mean']:.1f} (n={int(row['count'])})",
            (row["mean"], position),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#333333",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [workflow_display_name(workflow) for workflow in plot_summary["workflow"]]
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 21.8)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Raw NASA-TLX workload score (0-20)")
    ax.set_ylabel("Assigned workflow")
    ax.set_title("Raw NASA-TLX Workload by Workflow in Practice Rounds")
    apply_standard_axes_style(ax, grid_axis="x")

    save_figure(
        fig,
        slug,
        "Raw NASA-TLX Workload by Workflow in Practice Rounds",
        (
            "Raw NASA-TLX is the equally weighted mean of the six subscales for "
            "each practice round observation. Higher scores indicate greater "
            "perceived workload. Points show workflow means and whiskers show "
            "approximate 95% confidence intervals."
        ),
    )


# ---------------------------------------------------------------------------
# 34: Satisfaction versus external quality
# ---------------------------------------------------------------------------


def _spearman_summary(
    dataframe: pd.DataFrame,
    phase: str,
) -> dict[str, float | int | str]:
    """Return a descriptive pooled Spearman association for one study phase."""
    phase_data = dataframe[dataframe["phase"].eq(phase)].dropna(
        subset=[SATISFACTION_COLUMN, QUALITY_PRIMARY_METRIC]
    )

    rho = phase_data[SATISFACTION_COLUMN].corr(
        phase_data[QUALITY_PRIMARY_METRIC],
        method="spearman",
    )

    return {
        "phase": phase,
        "observations": int(len(phase_data)),
        "spearmanRho": float(rho) if pd.notna(rho) else np.nan,
        "meanSatisfaction": (
            float(phase_data[SATISFACTION_COLUMN].mean())
            if not phase_data.empty
            else np.nan
        ),
        "meanQuality": (
            float(phase_data[QUALITY_PRIMARY_METRIC].mean())
            if not phase_data.empty
            else np.nan
        ),
    }


def plot_satisfaction_vs_external_quality(prepared) -> None:
    """Show how external quality varies across participant satisfaction levels.

    Satisfaction is treated as an ordered 1-5 response scale. Within each
    study phase, the figure shows the distribution of external quality for
    each satisfaction level, together with raw participant-round observations
    and the median quality trend.
    """
    slug = "34_satisfaction_vs_external_quality"

    required_columns = {
        "phase",
        "workflow",
        "roundIndex",
        SATISFACTION_COLUMN,
        QUALITY_PRIMARY_METRIC,
    }
    if not require_columns(
        prepared,
        required_columns,
        "satisfaction versus external quality",
    ):
        return

    plot_df = prepared.copy()

    plot_df[SATISFACTION_COLUMN] = pd.to_numeric(
        plot_df[SATISFACTION_COLUMN],
        errors="coerce",
    )
    plot_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        plot_df[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            "phase",
            SATISFACTION_COLUMN,
            QUALITY_PRIMARY_METRIC,
        ]
    )

    plot_df = plot_df[
        plot_df[SATISFACTION_COLUMN].between(1, 5)
        & plot_df[QUALITY_PRIMARY_METRIC].between(
            QUALITY_SCALE_MIN,
            QUALITY_SCALE_MAX,
        )
    ].copy()

    if plot_df.empty:
        return

    available_phases = [
        phase for phase in PHASES if phase in set(plot_df["phase"].dropna())
    ]

    if not available_phases:
        return

    statistics = pd.DataFrame(
        [_spearman_summary(plot_df, phase) for phase in available_phases]
    )
    save_table(statistics, slug, index=False)

    satisfaction_levels = np.arange(1, 6)

    fig, axes = plt.subplots(
        1,
        len(available_phases),
        figsize=(7.0 * len(available_phases), 5.6),
        sharey=True,
        squeeze=False,
    )

    for phase_index, (axis, phase) in enumerate(zip(axes.flatten(), available_phases)):
        phase_df = plot_df[plot_df["phase"].eq(phase)].copy()

        box_data = []
        box_positions = []
        counts = []
        median_x = []
        median_y = []

        rng = np.random.default_rng(700 + phase_index)

        for satisfaction_level in satisfaction_levels:
            quality_values = phase_df.loc[
                phase_df[SATISFACTION_COLUMN].eq(satisfaction_level),
                QUALITY_PRIMARY_METRIC,
            ].dropna()

            counts.append(len(quality_values))

            if quality_values.empty:
                continue

            box_data.append(quality_values.to_numpy(dtype=float))
            box_positions.append(satisfaction_level)

            # Raw observations, horizontally jittered only to prevent overlap.
            x_jitter = satisfaction_level + rng.uniform(
                -0.16,
                0.16,
                size=len(quality_values),
            )

            axis.scatter(
                x_jitter,
                quality_values,
                s=26,
                color="#777777",
                alpha=0.38,
                edgecolor="none",
                zorder=2,
            )

            median_x.append(satisfaction_level)
            median_y.append(float(quality_values.median()))

        if box_data:
            boxplot = axis.boxplot(
                box_data,
                positions=box_positions,
                widths=0.56,
                patch_artist=True,
                showfliers=False,
                medianprops={
                    "color": "#222222",
                    "linewidth": 1.8,
                },
                boxprops={
                    "edgecolor": "#555555",
                    "linewidth": 1.1,
                },
                whiskerprops={
                    "color": "#555555",
                    "linewidth": 1.0,
                },
                capprops={
                    "color": "#555555",
                    "linewidth": 1.0,
                },
            )

            for box in boxplot["boxes"]:
                box.set_facecolor("#e6e6e6")
                box.set_alpha(0.95)

        # Connect medians to make the overall descriptive pattern easy to see.
        if len(median_x) >= 2:
            axis.plot(
                median_x,
                median_y,
                color="#222222",
                linewidth=1.6,
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="#222222",
                zorder=4,
            )

        phase_stat = statistics.loc[statistics["phase"].eq(phase)].iloc[0]

        rho = phase_stat.get("spearmanRho")
        observations = int(phase_stat.get("observations", 0))

        statistic_text = (
            f"Spearman ρ = {rho:.2f}\nn = {observations}"
            if pd.notna(rho)
            else f"n = {observations}"
        )

        axis.text(
            0.04,
            0.96,
            statistic_text,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d0d0d0",
                "alpha": 0.95,
            },
        )

        axis.set_title(
            f"{phase.capitalize()} rounds",
            fontsize=11,
            fontweight="bold",
        )

        axis.set_xlim(0.5, 5.5)
        axis.set_ylim(QUALITY_SCALE_MIN - 0.3, QUALITY_SCALE_MAX + 0.3)

        axis.set_xticks(satisfaction_levels)
        axis.set_xticklabels(
            [
                f"{level}\n(n={count})"
                for level, count in zip(satisfaction_levels, counts)
            ]
        )

        axis.set_yticks(
            np.arange(
                int(QUALITY_SCALE_MIN),
                int(QUALITY_SCALE_MAX) + 1,
            )
        )

        axis.set_xlabel("Participant satisfaction rating\n(n shown below each rating)")
        axis.set_ylabel("External quality composite (1-5)")

        apply_standard_axes_style(axis, grid_axis="y")

    raw_points_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#777777",
        markersize=7,
        alpha=0.55,
        label="Individual participant-round",
    )

    median_handle = plt.Line2D(
        [0],
        [0],
        color="#222222",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#222222",
        linewidth=1.6,
        markersize=5,
        label="Median external quality",
    )

    fig.legend(
        handles=[raw_points_handle, median_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "External Text Quality by Participant Satisfaction",
        fontsize=13,
        y=0.995,
    )

    fig.text(
        0.02,
        0.085,
        "Boxes show the interquartile range; horizontal lines inside boxes show "
        "the median. The connected median line is descriptive only. Spearman "
        "correlations pool workflows within each study phase.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4a4a4a",
    )

    fig.tight_layout(rect=(0, 0.13, 1, 0.97))

    save_figure(
        fig,
        slug,
        "External Text Quality by Participant Satisfaction",
        "Distribution of external quality-composite scores at each participant "
        "satisfaction level, shown separately for Practice and Main rounds. "
        "Boxes show interquartile ranges, individual points show participant-rounds, "
        "and connected markers show median external quality. Spearman correlations "
        "are descriptive associations pooling workflows within each study phase.",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_experience(df: pd.DataFrame) -> None:
    """Generate participant subjective-experience figures."""
    prepared = _prepare_experience_data(df)
    if prepared.empty:
        return

    practice_df = phase_data(prepared, "practice")
    if not practice_df.empty:
        plot_satisfaction_by_practice_round_and_workflow(practice_df)
        plot_ai_experience_by_practice_round_and_workflow(practice_df)
        plot_tlx_score_by_workflow_in_practice_rounds(practice_df)

    plot_satisfaction_vs_external_quality(prepared)
