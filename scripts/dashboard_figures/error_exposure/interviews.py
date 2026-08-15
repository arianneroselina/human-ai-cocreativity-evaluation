"""Interview-coded awareness and other reported AI issues."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    AWARENESS_LABELS,
    OTHER_AI_ERROR_LABELS,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    FOOTNOTE_TEXT_COLOR,
    apply_standard_axes_style, VALUE_LABEL_FONT_SIZE, SUBTITLE_FONT_SIZE,
)
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


def plot_injected_error_awareness(
    prepared: pd.DataFrame,
    interview_notes: pd.DataFrame | None = None,
) -> None:
    """Show whether exposed interview respondents noticed the injected error."""
    slug = "110_injected_error_awareness"

    notes = (
        interview_notes.copy()
        if interview_notes is not None
        else load_participant_interview_notes(prepared)
    )
    required = {"injectedErrorExperience"}
    if notes.empty or not require_columns(
        notes, required, "injected-error awareness notes"
    ):
        return

    exposed = notes[notes["errorExposed"]].copy()
    if exposed.empty:
        print(
            "Skipping Figure 110; no exposed interview respondents with "
            "awareness coding were available."
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
            fontsize=VALUE_LABEL_FONT_SIZE,
        )
    ax.set_xlim(0, 112)
    ax.set_xlabel("Exposed interview respondents (%)")
    ax.set_ylabel("")
    ax.set_title("Awareness of the Injected AI Error")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Awareness of the Injected AI Error",
        "Interview-coded awareness among respondents actually exposed to the "
        "injected error in Main Round 1.",
    )


def plot_other_ai_error_types(
    prepared: pd.DataFrame,
    interview_notes: pd.DataFrame | None = None,
) -> None:
    """Show non-injected AI issues reported by interview respondents."""
    slug = "111_reported_other_ai_error_types"

    notes = (
        interview_notes.copy()
        if interview_notes is not None
        else load_participant_interview_notes(prepared)
    )
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
            "Skipping Figure 111; no coded non-injected AI error types were available."
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
            fontsize=VALUE_LABEL_FONT_SIZE,
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
            f"reported at least one other AI issue.\n"
            f"There were {total_issue_reports} issue-type reports in total; "
            "participants could report multiple issue types."
        ),
        ha="left",
        va="bottom",
        fontsize=SUBTITLE_FONT_SIZE,
        color=FOOTNOTE_TEXT_COLOR,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    save_figure(
        fig,
        slug,
        "Other AI Error Types Reported in Interviews",
        "Interview-coded non-injected AI issues reported by participants. "
        "Categories are not mutually exclusive.",
    )
