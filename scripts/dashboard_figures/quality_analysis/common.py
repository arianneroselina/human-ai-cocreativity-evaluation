"""Shared quality preparation and plotting primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scripts.config import (
    QUALITY_PRIMARY_METRIC,
    QUALITY_Y_MAX,
    QUALITY_Y_MIN,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    quality_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import require_columns, save_figure, save_table


def _prepare_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid observations for the primary quality outcome."""
    if QUALITY_PRIMARY_METRIC not in df.columns:
        return pd.DataFrame()

    prepared = df.copy()
    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    return prepared.dropna(subset=[QUALITY_PRIMARY_METRIC]).copy()


def _workflow_order_present(dataframe: pd.DataFrame) -> list[str]:
    """Return available workflows in the canonical display order."""
    available = set(dataframe["workflow"].dropna().unique())
    return [workflow for workflow in WORKFLOW_ORDER if workflow in available]


def _add_raw_points(
    ax,
    dataframe: pd.DataFrame,
    workflows: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
    seed: int = 42,
) -> None:
    """Add reproducibly jittered poem-level observations."""
    rng = np.random.default_rng(seed)

    for position, workflow in enumerate(workflows, start=1):
        values = (
            dataframe.loc[dataframe["workflow"].eq(workflow), metric]
            .dropna()
            .to_numpy()
        )
        if len(values) == 0:
            continue

        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=WORKFLOW_COLORS[workflow],
            alpha=0.48,
            s=28,
            zorder=3,
            linewidths=0,
        )


def _add_mean_intervals(
    ax,
    summary: pd.DataFrame,
    workflows: list[str],
) -> None:
    """Overlay workflow means and descriptive 95% confidence intervals."""
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        if workflow not in indexed_summary.index:
            continue

        row = indexed_summary.loc[workflow]
        mean = float(row["mean"])
        low = row["ciLow"]
        high = row["ciHigh"]

        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color="black",
                markerfacecolor="white",
                markeredgewidth=1.2,
                capsize=4,
                zorder=5,
            )
        else:
            ax.scatter(
                position,
                mean,
                marker="D",
                color="black",
                facecolor="white",
                zorder=5,
            )


def _annotate_workflow_counts(
    ax,
    summary: pd.DataFrame,
    workflows: list[str],
    y_position: float = QUALITY_Y_MIN + 0.12,
) -> None:
    """Annotate workflow plots with observation counts."""
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        if workflow not in indexed_summary.index:
            continue

        ax.text(
            position,
            y_position,
            f"n={int(indexed_summary.loc[workflow, 'count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_workflow_quality_distribution(
    dataframe: pd.DataFrame,
    *,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Create a raw-point, boxplot, and mean/CI workflow comparison."""
    if dataframe.empty:
        return

    workflows = _workflow_order_present(dataframe)
    if not workflows:
        return

    summary = quality_summary(dataframe, ["workflow"])
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    save_table(summary, slug, index=False)

    box_data = [
        dataframe.loc[
            dataframe["workflow"].eq(workflow),
            QUALITY_PRIMARY_METRIC,
        ]
        .dropna()
        .to_numpy()
        for workflow in workflows
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    boxplot = ax.boxplot(
        box_data,
        tick_labels=[workflow_display_name(workflow) for workflow in workflows],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={"marker": "", "markersize": 0},
    )

    for patch, workflow in zip(boxplot["boxes"], workflows):
        patch.set_facecolor(WORKFLOW_COLORS[workflow])
        patch.set_alpha(0.35)
        patch.set_edgecolor(BAR_EDGE_COLOR)

    _add_raw_points(ax, dataframe, workflows)
    _add_mean_intervals(ax, summary, workflows)
    _annotate_workflow_counts(ax, summary, workflows)

    mean_handle = Line2D(
        [],
        [],
        marker="D",
        color="black",
        markerfacecolor="white",
        linestyle="None",
        label="Mean ± 95% CI",
    )
    ax.legend(handles=[mean_handle], loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Mean overall quality (1-5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)


def _paired_matrix(
    dataframe: pd.DataFrame,
    workflows: list[str],
) -> pd.DataFrame:
    """Build a complete participant-by-workflow matrix."""
    required = {"participantId", "workflow", QUALITY_PRIMARY_METRIC}
    if not require_columns(dataframe, required, "paired quality comparison"):
        return pd.DataFrame()

    paired_source = dataframe.loc[
        dataframe["workflow"].isin(workflows),
        ["participantId", "workflow", QUALITY_PRIMARY_METRIC],
    ].dropna()

    duplicate_mask = paired_source.duplicated(
        subset=["participantId", "workflow"],
        keep=False,
    )
    if duplicate_mask.any():
        print(
            "Paired quality comparison found duplicate participant-workflow "
            "observations; duplicate values are averaged."
        )

    matrix = (
        paired_source.groupby(
            ["participantId", "workflow"],
            observed=True,
        )[QUALITY_PRIMARY_METRIC]
        .mean()
        .unstack("workflow")
        .reindex(columns=workflows)
    )

    return matrix.dropna(subset=workflows)


def _plot_two_condition_paired_comparison(
    paired_df: pd.DataFrame,
    *,
    left_column: str,
    right_column: str,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Draw a within-participant two-condition quality comparison."""
    if paired_df.empty:
        return

    export_df = paired_df.reset_index(drop=True).copy()
    export_df.insert(
        0,
        "participantCode",
        [f"P{index:02d}" for index in range(1, len(export_df) + 1)],
    )
    export_df = export_df.rename(
        columns={
            left_column: left_label,
            right_column: right_label,
        }
    )
    save_table(export_df, slug, index=False)

    summary = quality_summary(
        paired_df[[left_column, right_column]].melt(
            var_name="comparison",
            value_name=QUALITY_PRIMARY_METRIC,
        ),
        ["comparison"],
    )
    save_table(summary, f"{slug}_summary", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    rng = np.random.default_rng(42)
    x_positions = np.array([1.0, 2.0])

    for _, row in paired_df.iterrows():
        jitter = rng.uniform(-0.035, 0.035, size=2)
        x_values = x_positions + jitter
        y_values = [row[left_column], row[right_column]]

        ax.plot(
            x_values,
            y_values,
            color="0.65",
            alpha=0.55,
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            x_values[0],
            y_values[0],
            color=left_color,
            alpha=0.75,
            s=30,
            zorder=2,
        )
        ax.scatter(
            x_values[1],
            y_values[1],
            color=right_color,
            alpha=0.75,
            s=30,
            zorder=2,
        )

    indexed_summary = summary.set_index("comparison")
    for x_position, comparison, color in zip(
        x_positions,
        [left_column, right_column],
        [left_color, right_color],
    ):
        row = indexed_summary.loc[comparison]
        mean = float(row["mean"])
        low = row["ciLow"]
        high = row["ciHigh"]

        if pd.notna(low) and pd.notna(high):
            ax.errorbar(
                x_position,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="D",
                color="black",
                markerfacecolor=color,
                markeredgecolor="black",
                capsize=4,
                markersize=7,
                zorder=5,
            )
        else:
            ax.scatter(
                x_position,
                mean,
                marker="D",
                color="black",
                facecolor=color,
                s=55,
                zorder=5,
            )

    ax.text(
        0.5,
        0.03,
        f"Paired participants: n={len(paired_df)}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([left_label, right_label])
    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.set_title(title)
    ax.set_ylabel("Mean overall quality (1-5)")
    apply_standard_axes_style(ax)

    save_figure(fig, slug, title, description)
