"""Injected AI-error exposure analysis.

41  Main Round 1 workflow choice and actual injected-error exposure
42  Workflow choices in Main Rounds 2-3 by Round-5 exposure group
43  Awareness of the injected error among exposed interview respondents
44  Other AI error types reported in interviews

Exposure is determined by the workflow voluntarily selected in Main Round 1.
Post-error comparisons are therefore descriptive, not randomized causal effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    ERROR_ROUND_INDEX,
    WORKFLOW_COLORS,
    WORKFLOW_ORDER,
    AWARENESS_LABELS,
    OTHER_AI_ERROR_LABELS,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    workflow_display_name,
    round_display_name,
    build_valid_ranking_rows,
    ranking_summary,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.plots_workflow import RANK_COLORS
from scripts.dashboard_figures.style import BAR_EDGE_COLOR, apply_standard_axes_style
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# 41: Workflow choice in Main Round 1
# ---------------------------------------------------------------------------


def plot_round5_workflow_choice(prepared) -> None:
    """Show voluntary workflow choices in the first Main round."""
    slug = "41_main_round1_workflow_choice"

    round5 = prepared[prepared["roundIndex"].eq(ERROR_ROUND_INDEX)].copy()
    if round5.empty:
        return

    summary = (
        round5.groupby("workflow")
        .size()
        .reindex(WORKFLOW_ORDER, fill_value=0)
        .rename("participantCount")
        .reset_index()
    )

    summary = summary[summary["participantCount"].gt(0)].copy()

    total_participants = int(summary["participantCount"].sum())
    if total_participants == 0:
        return

    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["percentage"] = summary["participantCount"] / total_participants * 100

    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    bars = ax.bar(
        summary["workflowLabel"],
        summary["participantCount"],
        color=[WORKFLOW_COLORS[workflow] for workflow in summary["workflow"]],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{int(row['participantCount'])}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, summary["participantCount"].max() + 5)
    ax.set_xlabel("Workflow selected")
    ax.set_ylabel("Participants")
    ax.set_title("Workflow Choices in Main Round 1")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    apply_standard_axes_style(ax, grid_axis="y")

    fig.text(
        0.01,
        0.01,
        (
            f"n = {total_participants}. AI-supported selections in this round "
            "were the opportunity for participants to encounter the injected error."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Workflow Choices in Main Round 1",
        "Voluntary workflow selections in the first Main round. Participants "
        "selecting an AI-supported workflow encountered the injected-error condition.",
    )


# ---------------------------------------------------------------------------
# 42: Workflow choices after the injected-error round
# ---------------------------------------------------------------------------


def plot_post_error_workflow_choices_by_exposure(prepared) -> None:
    """Show post-error Main Round 2-3 workflow distributions by Round-5 exposure."""
    slug = "42_post_error_workflow_choices_by_exposure"

    post = (
        prepared[prepared["roundIndex"].gt(ERROR_ROUND_INDEX)]
        .dropna(subset=["errorExposed"])
        .copy()
    )
    if post.empty:
        return

    groups = [group for group in [True, False] if group in set(post["errorExposed"])]
    rounds = sorted(post["roundIndex"].unique().tolist())
    grid = pd.MultiIndex.from_product(
        [groups, rounds, WORKFLOW_ORDER],
        names=["errorExposed", "roundIndex", "workflow"],
    )
    summary = (
        post.groupby(["errorExposed", "roundIndex", "workflow"])
        .size()
        .reindex(grid, fill_value=0)
        .rename("choiceCount")
        .reset_index()
    )
    summary["roundTotal"] = summary.groupby(["errorExposed", "roundIndex"])[
        "choiceCount"
    ].transform("sum")
    summary["choicePercentage"] = np.where(
        summary["roundTotal"] > 0,
        summary["choiceCount"] / summary["roundTotal"] * 100,
        np.nan,
    )
    summary["workflowLabel"] = summary["workflow"].map(workflow_display_name)
    summary["exposureLabel"] = summary["errorExposed"].map(exposure_display_name)
    summary["mainRoundLabel"] = summary["roundIndex"].map(round_display_name)
    save_table(summary, slug, index=False)

    fig, axes = plt.subplots(
        1, len(groups), figsize=(6.2 * len(groups), 5.5), sharey=True, squeeze=False
    )
    for ax, group in zip(axes.flatten(), groups):
        group_summary = summary[summary["errorExposed"].eq(group)]
        bottoms = np.zeros(len(rounds), dtype=float)

        for workflow in WORKFLOW_ORDER:
            values = (
                group_summary[group_summary["workflow"].eq(workflow)]
                .set_index("roundIndex")
                .reindex(rounds)
            )
            percentages = values["choicePercentage"].to_numpy(dtype=float)
            counts = values["choiceCount"].to_numpy(dtype=int)
            bars = ax.bar(
                np.arange(len(rounds)),
                percentages,
                bottom=bottoms,
                color=WORKFLOW_COLORS[workflow],
                edgecolor=BAR_EDGE_COLOR,
                linewidth=0.8,
                label=workflow_display_name(workflow),
                zorder=2,
            )
            for bar, percent, count, bottom in zip(bars, percentages, counts, bottoms):
                if percent >= 9:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + percent / 2,
                        f"{count}\n{percent:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="black",
                    )
            bottoms += percentages

        totals = (
            group_summary.drop_duplicates("roundIndex")
            .set_index("roundIndex")
            .reindex(rounds)["roundTotal"]
            .to_numpy(dtype=int)
        )
        ax.set_xticks(np.arange(len(rounds)))
        ax.set_xticklabels(
            [f"{round_display_name(r)}\nn={n}" for r, n in zip(rounds, totals)]
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("Free-choice post-error round")
        ax.set_title(exposure_display_name(group))
        apply_standard_axes_style(ax, grid_axis="y")

    axes[0, 0].set_ylabel("Workflow choices (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Workflow selected",
        bbox_to_anchor=(0.99, 0.5),
        loc="center left",
    )
    fig.suptitle(
        "Post-Error Workflow Choices by Main Round 1 Exposure", fontsize=13, y=0.99
    )
    fig.text(
        0.01,
        0.01,
        "Each bar contains all workflow choices in that round. Exposure was defined from Main Round 1 and was not independently randomized.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Main Round 1 Exposure",
        "Distribution of voluntary workflow choices in Main Rounds 2-3 by error exposure.",
    )


# ---------------------------------------------------------------------------
# 43: Workflow ranking by reported AI error
# ---------------------------------------------------------------------------


def _reported_ai_error_groups(
    round_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one reported-AI-error group per participant session."""
    notes = load_participant_interview_notes(round_df)

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
) -> None:
    """Show final workflow-ranking distributions by reported AI-error count."""
    slug = "43_final_workflow_preference_by_reported_ai_errors"

    error_groups = _reported_ai_error_groups(prepared)
    if ranking_rows.empty or error_groups.empty:
        return

    ranking_df = ranking_rows.copy()
    ranking_df["sessionId"] = ranking_df["sessionId"].astype(str)

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

    # Use the overall preference order consistently in every panel.
    overall_summary = ranking_summary(ranking_df)
    workflow_order = overall_summary["meanRank"].sort_values().index.tolist()

    group_sizes = {
        group: ranking_df.loc[
            ranking_df["errorGroup"].eq(group),
            "sessionId",
        ].nunique()
        for group in observed_groups
    }

    export_rows = []

    fig, axes = plt.subplots(
        1,
        len(observed_groups),
        figsize=(4.9 * len(observed_groups), 5.3),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    axes = axes.flatten()

    for ax, group in zip(axes, observed_groups):
        group_df = ranking_df[ranking_df["errorGroup"].eq(group)]
        participant_count = group_sizes[group]

        summary = ranking_summary(group_df).reindex(workflow_order)

        rank_counts = summary[rank_columns].fillna(0)
        row_totals = rank_counts.sum(axis=1)
        rank_percentages = rank_counts.div(row_totals, axis=0) * 100

        positions = np.arange(len(workflow_order))
        left = np.zeros(len(workflow_order))

        for rank in rank_columns:
            percentages = rank_percentages[rank].to_numpy()
            counts = rank_counts[rank].astype(int).to_numpy()

            bars = ax.barh(
                positions,
                percentages,
                left=left,
                label=f"Rank {rank}",
                color=RANK_COLORS[rank - 1],
                edgecolor=BAR_EDGE_COLOR,
            )

            # Show participant counts only in sufficiently wide segments.
            for index, (bar, percentage, count) in enumerate(
                zip(bars, percentages, counts)
            ):
                if percentage >= 18:
                    ax.text(
                        left[index] + percentage / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            left += percentages

        # Display the mean rank as a separate aligned value.
        for index, workflow in enumerate(workflow_order):
            mean_rank = summary.loc[workflow, "meanRank"]

            ax.text(
                102,
                index,
                f"{mean_rank:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )

            export_rows.append(
                {
                    "errorGroup": group,
                    "participants": participant_count,
                    "workflow": workflow_display_name(workflow),
                    "meanRank": mean_rank,
                    **{
                        f"Rank {rank} count": int(rank_counts.loc[workflow, rank])
                        for rank in rank_columns
                    },
                }
            )

        ax.set_title(
            f"{group}\n(n={participant_count})",
            fontsize=11,
        )
        ax.set_xlabel("Participants assigning each rank (%)")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])

        ax.set_yticks(positions)
        ax.set_yticklabels(
            [workflow_display_name(workflow) for workflow in workflow_order]
        )
        ax.invert_yaxis()

        apply_standard_axes_style(ax, grid_axis="x")

        # Header for the aligned mean-rank values.
        ax.text(
            102,
            -0.65,
            "Mean\nrank",
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.35",
            clip_on=False,
        )

    save_table(
        pd.DataFrame(export_rows),
        slug,
        index=False,
    )

    axes[-1].legend(
        title="Assigned rank",
        bbox_to_anchor=(1.08, 1),
        loc="upper left",
    )

    fig.suptitle(
        "Final Workflow Preference by Number of Reported AI Errors",
        fontsize=13,
    )

    save_figure(
        fig,
        slug,
        "Final Workflow Preference by Number of Reported AI Errors",
        (
            "Percentage distribution and mean rank of final workflow preferences, "
            "grouped by the number of AI errors participants reported noticing. "
            "Rank 1 indicates the strongest preference. Numbers inside sufficiently "
            "large segments show participant counts. The groups are descriptive; "
            "reporting no error does not necessarily indicate that no AI error "
            "occurred or was encountered."
        ),
    )


# ---------------------------------------------------------------------------
# 44: Interview awareness among exposed participants
# ---------------------------------------------------------------------------


def plot_injected_error_awareness(prepared) -> None:
    """Show whether exposed interview respondents noticed the injected error."""
    slug = "44_injected_error_awareness"

    notes = load_participant_interview_notes(prepared)
    required = {"injectedErrorExperience"}
    if notes.empty or not require_columns(
        notes, required, "injected-error awareness notes"
    ):
        return

    exposed = notes[notes["errorExposed"]].copy()
    if exposed.empty:
        print(
            "Skipping Figure 43; no exposed interview respondents with awareness coding were available."
        )
        return

    summary = (
        exposed["injectedErrorExperience"]
        .value_counts()
        .reindex(["noticed", "not_noticed"], fill_value=0)
        .rename_axis("awarenessCode")
        .reset_index(name="participantCount")
    )
    denominator = int(summary["participantCount"].sum())
    summary["awarenessLabel"] = summary["awarenessCode"].map(AWARENESS_LABELS)
    summary["percentage"] = summary["participantCount"] / denominator * 100
    summary["interviewRespondentsTotal"] = int(notes["participantId"].nunique())
    summary["exposedInterviewRespondents"] = denominator
    save_table(summary, slug, index=False)

    plot_df = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.barh(
        plot_df["awarenessLabel"],
        plot_df["percentage"],
        edgecolor=BAR_EDGE_COLOR,
    )
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['participantCount'])}/{denominator} ({row['percentage']:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 112)
    ax.set_xlabel("Exposed interview respondents (%)")
    ax.set_ylabel("")
    ax.set_title("Awareness of the Injected AI Error")
    apply_standard_axes_style(ax, grid_axis="x")
    fig.text(
        0.01,
        0.01,
        f"Denominator: {denominator} interview respondent(s) confirmed as error-exposed in Main Round 1. Non-exposed respondents are excluded.",
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Awareness of the Injected AI Error",
        "Interview-coded awareness among respondents actually exposed to the injected error in Main Round 1.",
    )


# ---------------------------------------------------------------------------
# 45: Other AI error types reported in interviews
# ---------------------------------------------------------------------------


def plot_other_ai_error_types(prepared) -> None:
    """Show non-injected AI issues reported by interview respondents."""
    slug = "45_reported_other_ai_error_types"

    notes = load_participant_interview_notes(prepared)
    required = {"reportedOtherAiErrorTypes"}
    if notes.empty or not require_columns(
        notes, required, "other AI error interview notes"
    ):
        return

    total_respondents = int(notes["participantId"].nunique())
    rows = []
    for _, row in notes.iterrows():
        raw_types = row.get("reportedOtherAiErrorTypes")
        if pd.isna(raw_types) or not str(raw_types).strip():
            continue
        for raw_type in str(raw_types).split(";"):
            error_type = raw_type.strip()
            if error_type:
                rows.append(
                    {"participantId": row["participantId"], "errorType": error_type}
                )
    if not rows:
        print(
            "Skipping Figure 44; no coded non-injected AI error types were available."
        )
        return

    error_type_df = pd.DataFrame(rows).drop_duplicates(
        subset=["participantId", "errorType"]
    )

    unique_reporters = error_type_df["participantId"].nunique()
    total_issue_reports = len(error_type_df)

    summary = (
        pd.DataFrame(rows)
        .groupby("errorType")["participantId"]
        .nunique()
        .reset_index(name="participantCount")
    )
    summary["errorTypeLabel"] = (
        summary["errorType"]
        .map(OTHER_AI_ERROR_LABELS)
        .fillna(summary["errorType"].str.replace("_", " ").str.title())
    )
    summary["percentage"] = summary["participantCount"] / total_respondents * 100
    summary = summary.sort_values(
        ["participantCount", "errorTypeLabel"], ascending=[True, True]
    ).reset_index(drop=True)
    summary["interviewRespondentsTotal"] = total_respondents
    save_table(summary, slug, index=False)

    fig, ax = plt.subplots(figsize=(9.4, max(4.5, 0.65 * len(summary) + 2.4)))
    bars = ax.barh(
        summary["errorTypeLabel"],
        summary["participantCount"],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.8,
        zorder=2,
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['participantCount'])} ({row['percentage']:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, max(1, summary["participantCount"].max() + 1.8))
    ax.set_xlabel("Interview respondents reporting the issue")
    ax.set_ylabel("")
    ax.set_title("Other AI Error Types Reported in Interviews")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    apply_standard_axes_style(ax, grid_axis="x")
    fig.text(
        0.01,
        0.01,
        (
            f"{unique_reporters} of {total_respondents} interview respondents "
            f"reported at least one other AI issue. "
            f"There were {total_issue_reports} issue-type reports in total; "
            "participants could report multiple issue types."
        ),
        ha="left",
        va="bottom",
        fontsize=8.3,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Other AI Error Types Reported in Interviews",
        "Interview-coded non-injected AI issues reported by participants. Categories are not mutually exclusive.",
    )


def plot_error_exposure(df, feedback_df) -> None:
    """Generate injected-error exposure and interview-coding figures."""
    required = {"participantId", "roundIndex", "workflow", "errorExposed"}

    if df.empty or not require_columns(
        df,
        required,
        "error-exposure data",
    ):
        return

    prepared = df.copy()
    prepared["participantId"] = prepared["participantId"].astype(str)

    if prepared.empty:
        return

    ranking_rows, _ = build_valid_ranking_rows(feedback_df)

    plot_round5_workflow_choice(prepared)
    plot_post_error_workflow_choices_by_exposure(prepared)
    plot_final_workflow_preference_by_reported_ai_errors(ranking_rows, df)
    plot_injected_error_awareness(prepared)
    plot_other_ai_error_types(prepared)
