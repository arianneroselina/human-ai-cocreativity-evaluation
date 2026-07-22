"""Practice round workflow-efficiency figures.

Efficiency is presented through completion time and quality together rather
than through a quality-per-minute ratio.

Figures
-------
16  Total completion time by workflow in practice rounds
17  Quality-time efficiency profile by workflow in practice rounds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from scripts.config import (
    QUALITY_PRIMARY_METRIC,
    QUALITY_Y_MIN,
    QUALITY_Y_MAX,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
    CI_Z_VALUE,
)
from scripts.dashboard_figures.helpers import (
    phase_data,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def _add_total_completion_time(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, str] | tuple[pd.DataFrame, None]:
    """Add total completion time in minutes, including pauses where available.

    The helper prefers explicit elapsed or total-duration columns. It never uses
    ``effectiveTimeMinutes`` alone because that measure may exclude pauses.
    """
    prepared = dataframe.copy()

    if "totalCompletionTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["totalCompletionTimeMinutes"],
            errors="coerce",
        )
        return prepared, "totalCompletionTimeMinutes"

    if "elapsedTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["elapsedTimeMinutes"],
            errors="coerce",
        )
        return prepared, "elapsedTimeMinutes"

    if "totalTimeMinutes" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["totalTimeMinutes"],
            errors="coerce",
        )
        return prepared, "totalTimeMinutes"

    if {"effectiveTimeMinutes", "pauseTimeMinutes"}.issubset(prepared.columns):
        prepared["totalCompletionTimeMinutes"] = pd.to_numeric(
            prepared["effectiveTimeMinutes"], errors="coerce"
        ) + pd.to_numeric(prepared["pauseTimeMinutes"], errors="coerce")
        return prepared, "effectiveTimeMinutes + pauseTimeMinutes"

    if {"effectiveTimeMs", "pauseTimeMs"}.issubset(prepared.columns):
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["effectiveTimeMs"], errors="coerce")
            + pd.to_numeric(prepared["pauseTimeMs"], errors="coerce")
        ) / 60000
        return prepared, "effectiveTimeMs + pauseTimeMs"

    if "elapsedTimeMs" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["elapsedTimeMs"], errors="coerce") / 60000
        )
        return prepared, "elapsedTimeMs"

    if {"startedAt", "submittedAt"}.issubset(prepared.columns):
        started = pd.to_datetime(prepared["startedAt"], errors="coerce", utc=True)
        submitted = pd.to_datetime(
            prepared["submittedAt"],
            errors="coerce",
            utc=True,
        )
        prepared["totalCompletionTimeMinutes"] = (
            submitted - started
        ).dt.total_seconds() / 60
        return prepared, "submittedAt - startedAt"

    if "timeMs" in prepared.columns:
        prepared["totalCompletionTimeMinutes"] = (
            pd.to_numeric(prepared["timeMs"], errors="coerce") / 60000
        )
        return prepared, "timeMs"

    print(
        "Skipping efficiency figures; no total or elapsed completion-time "
        "measure was found. Effective time alone is not used because pauses "
        "must be included."
    )
    return pd.DataFrame(), None


def _prepare_efficiency_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Return valid practice-round quality and total-time observations."""
    required = {"workflow", QUALITY_PRIMARY_METRIC}
    if not require_columns(df, required, "practice-round efficiency"):
        return pd.DataFrame(), None

    practice_df = phase_data(df, "practice")
    if practice_df.empty:
        return pd.DataFrame(), None

    practice_df[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        practice_df[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )

    practice_df, time_source = _add_total_completion_time(practice_df)
    if practice_df.empty:
        return pd.DataFrame(), None

    practice_df = practice_df.dropna(
        subset=[
            "workflow",
            QUALITY_PRIMARY_METRIC,
            "totalCompletionTimeMinutes",
        ]
    )
    practice_df = practice_df.loc[
        practice_df["workflow"].isin(WORKFLOW_ORDER)
        & practice_df["totalCompletionTimeMinutes"].gt(0)
    ].copy()

    return practice_df, time_source


def _workflow_order_present(dataframe: pd.DataFrame) -> list[str]:
    """Return available workflows in canonical display order."""
    available = set(dataframe["workflow"].dropna().unique())
    return [workflow for workflow in WORKFLOW_ORDER if workflow in available]


def _mean_ci(values: pd.Series) -> dict[str, float | int]:
    """Return a descriptive mean and normal-approximation 95% CI."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = int(len(numeric))

    if count == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "ciLow": np.nan,
            "ciHigh": np.nan,
        }

    mean = float(numeric.mean())
    median = float(numeric.median())
    std = float(numeric.std(ddof=1)) if count > 1 else np.nan

    if count > 1 and np.isfinite(std):
        margin = CI_Z_VALUE * std / np.sqrt(count)
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "count": count,
        "mean": mean,
        "median": median,
        "std": std,
        "ciLow": ci_low,
        "ciHigh": ci_high,
    }


def _workflow_efficiency_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise completion time and quality for each workflow."""
    rows = []

    for workflow in _workflow_order_present(dataframe):
        workflow_df = dataframe.loc[dataframe["workflow"].eq(workflow)]
        time_summary = _mean_ci(workflow_df["totalCompletionTimeMinutes"])
        quality_summary = _mean_ci(workflow_df[QUALITY_PRIMARY_METRIC])

        rows.append(
            {
                "workflow": workflow,
                "workflowLabel": workflow_display_name(workflow),
                "count": min(
                    int(time_summary["count"]),
                    int(quality_summary["count"]),
                ),
                "meanCompletionTimeMinutes": time_summary["mean"],
                "medianCompletionTimeMinutes": time_summary["median"],
                "completionTimeStd": time_summary["std"],
                "completionTimeCiLow": max(
                    0.0,
                    float(time_summary["ciLow"]),
                )
                if pd.notna(time_summary["ciLow"])
                else np.nan,
                "completionTimeCiHigh": time_summary["ciHigh"],
                "meanQuality": quality_summary["mean"],
                "medianQuality": quality_summary["median"],
                "qualityStd": quality_summary["std"],
                "qualityCiLow": quality_summary["ciLow"],
                "qualityCiHigh": quality_summary["ciHigh"],
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 16: Total completion time
# -----------------------------------------------------------------------------


def plot_completion_time_by_workflow_practice_rounds(
    practice_df: pd.DataFrame,
    time_source: str,
) -> None:
    """Show total completion-time distributions for each practice workflow."""
    slug = "16_completion_time_by_workflow_practice_rounds"
    workflows = _workflow_order_present(practice_df)
    if not workflows:
        return

    summary = _workflow_efficiency_summary(practice_df)
    save_table(summary, slug, index=False)

    box_data = [
        practice_df.loc[
            practice_df["workflow"].eq(workflow),
            "totalCompletionTimeMinutes",
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

    rng = np.random.default_rng(42)
    indexed_summary = summary.set_index("workflow")

    for position, workflow in enumerate(workflows, start=1):
        values = (
            practice_df.loc[
                practice_df["workflow"].eq(workflow),
                "totalCompletionTimeMinutes",
            ]
            .dropna()
            .to_numpy()
        )
        jitter = rng.uniform(-0.13, 0.13, size=len(values))

        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=WORKFLOW_COLORS[workflow],
            alpha=0.48,
            s=28,
            linewidths=0,
            zorder=3,
        )

        row = indexed_summary.loc[workflow]
        mean = float(row["meanCompletionTimeMinutes"])
        low = row["completionTimeCiLow"]
        high = row["completionTimeCiHigh"]

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

        ax.text(
            position,
            0.02,
            f"n={int(row['count'])}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
        )

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
    ax.set_title("Total Completion Time by Workflow in Practice Rounds")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Total completion time including pauses (minutes)")
    ax.set_ylim(bottom=0)
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Total Completion Time by Workflow in Practice Rounds",
        (
            "Practice round completion times include pauses. Points show outputs, "
            "boxes show distributions, and diamonds show descriptive means with "
            f"95% confidence intervals."
        ),
    )


# -----------------------------------------------------------------------------
# 17: Combined quality-time efficiency profile
# -----------------------------------------------------------------------------


def plot_quality_time_efficiency_profile_practice_rounds(
    practice_df: pd.DataFrame,
    time_source: str,
) -> None:
    """Show workflow mean quality against mean total completion time."""
    slug = "17_quality_time_efficiency_profile_practice_rounds"
    summary = _workflow_efficiency_summary(practice_df)
    if summary.empty:
        return

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    annotation_offsets = {
        "human": (-72, -22),
        "ai": (-20, 20),
        "human_ai": (12, -24),
        "ai_human": (12, 20),
    }

    for _, row in summary.iterrows():
        workflow = row["workflow"]
        mean_time = float(row["meanCompletionTimeMinutes"])
        mean_quality = float(row["meanQuality"])

        x_low = row["completionTimeCiLow"]
        x_high = row["completionTimeCiHigh"]
        y_low = row["qualityCiLow"]
        y_high = row["qualityCiHigh"]

        xerr = None
        if pd.notna(x_low) and pd.notna(x_high):
            xerr = [[mean_time - x_low], [x_high - mean_time]]

        yerr = None
        if pd.notna(y_low) and pd.notna(y_high):
            yerr = [[mean_quality - y_low], [y_high - mean_quality]]

        ax.errorbar(
            mean_time,
            mean_quality,
            xerr=xerr,
            yerr=yerr,
            fmt="D",
            markersize=10,
            color=WORKFLOW_COLORS[workflow],
            markeredgecolor="black",
            markeredgewidth=1.0,
            capsize=4,
            linewidth=1.2,
            zorder=3,
        )

        ax.annotate(
            (
                f"{workflow_display_name(workflow)}\n"
                f"{mean_time:.2f} min · {mean_quality:.2f}/5"
            ),
            xy=(mean_time, mean_quality),
            xytext=annotation_offsets.get(workflow, (8, 8)),
            textcoords="offset points",
            fontsize=8.5,
            va="center",
        )

    ax.annotate(
        "Preferred direction:\nhigher quality, less time",
        xy=(0.05, 0.92),
        xytext=(0.23, 0.78),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.1,
            "color": "0.35",
        },
        ha="center",
        fontsize=8.5,
        color="0.30",
    )

    ax.set_title("Quality-Time Efficiency Profile in Practice Rounds")
    ax.set_xlabel("Mean total completion time including pauses (minutes)")
    ax.set_ylabel("Mean overall quality (1-5)")
    ax.set_ylim(QUALITY_Y_MIN, QUALITY_Y_MAX)
    ax.set_xlim(left=0)
    apply_standard_axes_style(ax)

    save_figure(
        fig,
        slug,
        "Quality-Time Efficiency Profile in Practice Rounds",
        (
            "Diamonds show workflow means; horizontal and vertical error bars show "
            "descriptive 95% confidence intervals for total completion time and "
            "quality. Workflows nearer the upper-left combine higher quality with "
            f"shorter completion time."
        ),
    )


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------


def plot_efficiency(df: pd.DataFrame) -> None:
    """Generate descriptive practice-round workflow-efficiency figures."""
    practice_df, time_source = _prepare_efficiency_data(df)
    if practice_df.empty or time_source is None:
        return

    plot_completion_time_by_workflow_practice_rounds(practice_df, time_source)
    plot_quality_time_efficiency_profile_practice_rounds(practice_df, time_source)
