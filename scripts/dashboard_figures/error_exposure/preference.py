"""Final workflow preference by reported AI-error count."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import WORKFLOW_ORDER
from scripts.dashboard_figures.helpers import (
    ranking_summary,
    workflow_display_name,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    save_figure,
    save_table,
)


def _reported_ai_error_groups(
    round_df: pd.DataFrame,
    interview_notes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one reported-AI-error group per participant session."""
    notes = (
        interview_notes.copy()
        if interview_notes is not None
        else load_participant_interview_notes(round_df)
    )

    if notes.empty or "sessionId" not in notes.columns:
        return pd.DataFrame()

    notes = notes.dropna(subset=["sessionId"]).copy()
    notes["sessionId"] = notes["sessionId"].astype(str)

    injected_error_count = (
        notes["injectedErrorExperience"].eq("noticed").astype(int)
        if "injectedErrorExperience" in notes.columns
        else pd.Series(0, index=notes.index, dtype=int)
    )

    if "reportedOtherAiErrorTypes" in notes.columns:
        other_error_counts = (
            notes[["participantId", "reportedOtherAiErrorTypes"]]
            .dropna(subset=["reportedOtherAiErrorTypes"])
            .assign(
                errorType=lambda data: (
                    data["reportedOtherAiErrorTypes"]
                    .astype("string")
                    .str.lower()
                    .str.split(";")
                )
            )
            .explode("errorType")
            .assign(errorType=lambda data: data["errorType"].str.strip())
            .loc[lambda data: data["errorType"].notna() & data["errorType"].ne("")]
            .groupby("participantId")["errorType"]
            .nunique()
        )

        notes["otherAiErrorCount"] = (
            notes["participantId"].map(other_error_counts).fillna(0).astype(int)
        )
    else:
        notes["otherAiErrorCount"] = 0

    notes["reportedAiErrorCount"] = injected_error_count + notes["otherAiErrorCount"]

    notes["errorGroup"] = pd.cut(
        notes["reportedAiErrorCount"],
        bins=[-1, 0, 1, np.inf],
        labels=[
            "No reported AI errors",
            "1 reported AI error",
            "2+ reported AI errors",
        ],
    )

    return notes[
        [
            "sessionId",
            "reportedAiErrorCount",
            "errorGroup",
        ]
    ].dropna(subset=["errorGroup"])


def plot_final_workflow_preference_by_reported_ai_errors(
    ranking_rows: pd.DataFrame,
    prepared: pd.DataFrame,
    interview_notes: pd.DataFrame | None = None,
) -> None:
    """Compare final workflow preferences by reported AI-error count.

    The left panel shows average assigned rank. The right panel shows the
    percentage of participants who ranked each workflow first.

    Lower mean ranks indicate stronger preference. Reported-error groups are
    descriptive and do not necessarily represent actual error exposure.
    """
    from matplotlib.lines import Line2D

    slug = "109_final_workflow_preference_by_reported_ai_errors"

    error_groups = _reported_ai_error_groups(prepared, interview_notes)

    if ranking_rows.empty or error_groups.empty:
        return

    ranking_df = ranking_rows.copy()
    ranking_df["sessionId"] = ranking_df["sessionId"].astype(str)

    error_groups = error_groups.copy()
    error_groups["sessionId"] = error_groups["sessionId"].astype(str)

    ranking_df = ranking_df.merge(
        error_groups,
        on="sessionId",
        how="inner",
        validate="many_to_one",
    )

    if ranking_df.empty:
        return

    group_order = [
        "No reported AI errors",
        "1 reported AI error",
        "2+ reported AI errors",
    ]

    observed_groups = [
        group for group in group_order if group in set(ranking_df["errorGroup"])
    ]

    if not observed_groups:
        return

    rank_columns = list(range(1, len(WORKFLOW_ORDER) + 1))

    # Keep the workflow order identical in both panels and all groups.
    overall_summary = ranking_summary(ranking_df)

    workflow_order = overall_summary["meanRank"].sort_values().index.tolist()

    if not workflow_order:
        return

    group_sizes = {
        group: ranking_df.loc[
            ranking_df["errorGroup"].eq(group),
            "sessionId",
        ].nunique()
        for group in observed_groups
    }

    group_colors = {
        "No reported AI errors": "#4C78A8",
        "1 reported AI error": "#F58518",
        "2+ reported AI errors": "#B279A2",
    }

    group_markers = {
        "No reported AI errors": "o",
        "1 reported AI error": "s",
        "2+ reported AI errors": "^",
    }

    group_offsets = dict(
        zip(
            observed_groups,
            np.linspace(
                -0.20,
                0.20,
                len(observed_groups),
            ),
        )
    )

    group_summaries = {}
    export_rows = []

    for group in observed_groups:
        group_df = ranking_df.loc[ranking_df["errorGroup"].eq(group)]

        summary = ranking_summary(group_df).reindex(workflow_order)

        rank_counts = (
            summary[rank_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        row_totals = rank_counts.sum(axis=1)

        first_choice_percent = (
            rank_counts[1].div(row_totals.replace(0, np.nan)).mul(100)
        )

        summary = summary.copy()
        summary["firstChoicePercent"] = first_choice_percent
        summary["rankingCount"] = row_totals

        group_summaries[group] = summary

        for workflow in workflow_order:
            mean_rank = summary.loc[workflow, "meanRank"]
            ranking_count = summary.loc[
                workflow,
                "rankingCount",
            ]
            first_choice_percentage = summary.loc[
                workflow,
                "firstChoicePercent",
            ]

            row = {
                "errorGroup": group,
                "groupParticipants": group_sizes[group],
                "workflow": workflow_display_name(workflow),
                "meanRank": mean_rank,
                "firstChoicePercent": first_choice_percentage,
                "validRankings": int(ranking_count),
            }

            for rank in rank_columns:
                count = int(rank_counts.loc[workflow, rank])

                percentage = (
                    count / ranking_count * 100 if ranking_count > 0 else np.nan
                )

                row[f"Rank {rank} count"] = count
                row[f"Rank {rank} percent"] = percentage

            export_rows.append(row)

    save_table(
        pd.DataFrame(export_rows),
        slug,
        index=False,
    )

    positions = np.arange(len(workflow_order))

    fig, (ax_mean, ax_first) = plt.subplots(
        ncols=2,
        figsize=(12.4, 5.5),
        sharey=True,
        gridspec_kw={
            "width_ratios": [1.15, 1.35],
        },
    )

    # ------------------------------------------------------------
    # Left panel: average preference rank
    # ------------------------------------------------------------
    for group in observed_groups:
        summary = group_summaries[group]
        offset = group_offsets[group]
        color = group_colors[group]
        marker = group_markers[group]

        mean_ranks = pd.to_numeric(
            summary["meanRank"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(mean_ranks)
        y_values = positions + offset

        # Subtle lollipop stems, without implying a trajectory.
        for y_value, mean_rank in zip(
            y_values[valid],
            mean_ranks[valid],
        ):
            ax_mean.hlines(
                y=y_value,
                xmin=1,
                xmax=mean_rank,
                color=color,
                linewidth=1.2,
                alpha=0.35,
                zorder=1,
            )

        ax_mean.scatter(
            mean_ranks[valid],
            y_values[valid],
            s=78,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

        for mean_rank, y_value in zip(
            mean_ranks[valid],
            y_values[valid],
        ):
            ax_mean.annotate(
                f"{mean_rank:.2f}",
                (mean_rank, y_value),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                color=color,
            )

    ax_mean.set_yticks(positions)
    ax_mean.set_yticklabels(
        [
            (f"{index + 1}. {workflow_display_name(workflow)}")
            for index, workflow in enumerate(workflow_order)
        ]
    )
    ax_mean.invert_yaxis()

    ax_mean.set_xlim(0.8, 4.45)
    ax_mean.set_xticks([1, 2, 3, 4])
    ax_mean.set_xticklabels(
        [
            "1\nBest",
            "2",
            "3",
            "4\nWorst",
        ]
    )

    ax_mean.set_xlabel("Average assigned rank")
    ax_mean.set_title("Average preference rank")

    apply_standard_axes_style(
        ax_mean,
        grid_axis="x",
    )

    # ------------------------------------------------------------
    # Right panel: first-choice percentage
    # ------------------------------------------------------------
    bar_height = 0.16

    for group in observed_groups:
        summary = group_summaries[group]
        offset = group_offsets[group]
        color = group_colors[group]

        percentages = pd.to_numeric(
            summary["firstChoicePercent"],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = np.isfinite(percentages)
        y_values = positions + offset

        bars = ax_first.barh(
            y_values[valid],
            percentages[valid],
            height=bar_height,
            color=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.7,
            zorder=2,
        )

        for bar, percentage in zip(
            bars,
            percentages[valid],
        ):
            y_center = bar.get_y() + bar.get_height() / 2

            if percentage >= 14:
                x_position = percentage - 2
                horizontal_alignment = "right"
                text_color = "white"
            else:
                x_position = percentage + 2
                horizontal_alignment = "left"
                text_color = color

            ax_first.text(
                x_position,
                y_center,
                f"{percentage:.0f}%",
                ha=horizontal_alignment,
                va="center",
                fontsize=8,
                fontweight="semibold",
                color=text_color,
                zorder=3,
            )

    ax_first.set_xlim(0, 105)
    ax_first.set_xticks([0, 25, 50, 75, 100])
    ax_first.set_xlabel("Participants ranking the workflow first (%)")
    ax_first.set_title("First-choice share")

    # Y labels are already displayed in the left panel.
    ax_first.tick_params(
        axis="y",
        labelleft=False,
        length=0,
    )

    apply_standard_axes_style(
        ax_first,
        grid_axis="x",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=group_markers[group],
            linestyle="none",
            markersize=8,
            markerfacecolor=group_colors[group],
            markeredgecolor="white",
            label=(f"{group} (n={group_sizes[group]})"),
        )
        for group in observed_groups
    ]

    fig.legend(
        handles=legend_handles,
        title="Reported AI errors",
        bbox_to_anchor=(0.85, 0.5),
        loc="center left",
    )

    total_participants = ranking_df["sessionId"].nunique()

    fig.suptitle(
        (f"Final Workflow Preference by Reported AI Errors (N={total_participants})"),
        fontsize=14,
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0.08, 0.82, 0.94))

    save_figure(
        fig,
        slug,
        "Final Workflow Preference by Reported AI Errors",
        (
            "Average final workflow rank and first-choice share, separated by "
            "the number of AI errors participants reported. Lower mean ranks "
            "represent stronger preference. The full counts and percentages "
            "for every assigned rank are retained in the exported table."
        ),
    )
