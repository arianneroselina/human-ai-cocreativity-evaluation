"""Post-error workflow choices by first Main-round exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    WORKFLOW_ORDER, AWARENESS_LABELS,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    round_display_name,
    workflow_display_name,
)
from scripts.dashboard_figures.loaders import load_participant_interview_notes
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    BAR_EDGE_COLOR,
    apply_standard_axes_style, VALUE_LABEL_FONT_SIZE,
)
from scripts.utils import (
    save_figure,
    save_table, require_columns,
)


def plot_post_error_workflow_choices_by_exposure(prepared) -> None:
    """Show post-error Main Round 2-3 workflow distributions by Round-5 exposure."""
    slug = "105_post_error_workflow_choices_by_exposure"

    post = (
        prepared[prepared["roundIndex"].gt(INJECTED_ERROR_ROUND_INDEX)]
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
                        fontsize=VALUE_LABEL_FONT_SIZE,
                        color="black",
                    )
            bottoms += percentages

            group_n = post.loc[
                post["errorExposed"].eq(group),
                "sessionId",
            ].nunique()

            ax.set_xticks(np.arange(len(rounds)))
            ax.set_xticklabels(
                [round_display_name(r) for r in rounds]
            )

            ax.set_ylim(0, 100)
            ax.set_xlabel("Post-error main round")
            ax.set_title(
                f"{exposure_display_name(group)} (n={group_n})"
            )
        apply_standard_axes_style(ax, grid_axis="y")

    axes[0, 0].set_ylabel("Workflow choices (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Workflow selected",
        bbox_to_anchor=(0.87, 0.5),
        loc="center left",
    )
    fig.suptitle(
        "Post-Error Workflow Choices by Main Round 1 Exposure", y=0.99
    )

    fig.tight_layout(rect=(0, 0.045, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Main Round 1 Exposure",
        "Distribution of voluntary workflow choices in Main Rounds 2-3 by "
        "error exposure.",
    )

def plot_post_error_workflow_choices_by_awareness(
        prepared: pd.DataFrame,
        interview_notes: pd.DataFrame | None = None,
) -> None:
    """Show post-error workflow choices among exposed participants by awareness."""
    slug = "112_post_error_workflow_choices_by_awareness"

    notes = (
        interview_notes.copy()
        if interview_notes is not None
        else load_participant_interview_notes(prepared)
    )

    required_note_columns = {
        "participantId",
        "errorExposed",
        "injectedErrorExperience",
    }
    if notes.empty or not require_columns(
            notes,
            required_note_columns,
            "post-error workflow choices by awareness",
    ):
        return

    # Only participants who were actually exposed to the injected error
    awareness = notes[
        notes["errorExposed"]
        & notes["injectedErrorExperience"].isin(["noticed", "not_noticed"])
        ][
        ["participantId", "injectedErrorExperience"]
    ].drop_duplicates(subset=["participantId"])

    if awareness.empty:
        print(
            "Skipping awareness analysis; no exposed participants with "
            "awareness coding were available."
        )
        return

    awareness = awareness.rename(
        columns={"injectedErrorExperience": "awarenessCode"}
    )
    awareness["awarenessLabel"] = awareness["awarenessCode"].map(
        AWARENESS_LABELS
    )

    # Only rounds after the injected-error round
    post = prepared[
        prepared["roundIndex"].gt(INJECTED_ERROR_ROUND_INDEX)
    ].copy()

    if post.empty:
        return

    # Join interview-coded awareness to round-level data
    post = post.merge(
        awareness,
        on="participantId",
        how="inner",
        validate="many_to_one",
    )

    if post.empty:
        print(
            "Skipping awareness analysis; no post-error rounds matched "
            "participants with awareness coding."
        )
        return

    awareness_order = ["noticed", "not_noticed"]
    groups = [
        group
        for group in awareness_order
        if group in set(post["awarenessCode"])
    ]
    rounds = sorted(post["roundIndex"].unique().tolist())

    # ------------------------------------------------------------------
    # Detailed workflow distribution
    # ------------------------------------------------------------------
    grid = pd.MultiIndex.from_product(
        [groups, rounds, WORKFLOW_ORDER],
        names=["awarenessCode", "roundIndex", "workflow"],
    )

    summary = (
        post.groupby(["awarenessCode", "roundIndex", "workflow"])
        .size()
        .reindex(grid, fill_value=0)
        .rename("choiceCount")
        .reset_index()
    )

    summary["roundTotal"] = summary.groupby(
        ["awarenessCode", "roundIndex"]
    )["choiceCount"].transform("sum")

    summary["choicePercentage"] = np.where(
        summary["roundTotal"] > 0,
        summary["choiceCount"] / summary["roundTotal"] * 100,
        np.nan,
        )

    summary["workflowLabel"] = summary["workflow"].map(
        workflow_display_name
    )
    summary["awarenessLabel"] = summary["awarenessCode"].map(
        AWARENESS_LABELS
    )
    summary["mainRoundLabel"] = summary["roundIndex"].map(
        round_display_name
    )

    # ------------------------------------------------------------------
    # Compact AI-supported vs Human-only summary
    # ------------------------------------------------------------------
    # Assumes Human-only is the only workflow without AI support.
    # Change this comparison if your internal workflow code is different.
    human_workflow = WORKFLOW_ORDER[0]

    post["aiSupported"] = post["workflow"].ne(human_workflow)

    ai_summary = (
        post.groupby(["awarenessCode", "roundIndex"])
        .agg(
            participantCount=("participantId", "nunique"),
            aiSupportedCount=("aiSupported", "sum"),
        )
        .reset_index()
    )

    ai_summary["aiSupportedPercentage"] = (
            ai_summary["aiSupportedCount"]
            / ai_summary["participantCount"]
            * 100
    )
    ai_summary["awarenessLabel"] = ai_summary["awarenessCode"].map(
        AWARENESS_LABELS
    )
    ai_summary["mainRoundLabel"] = ai_summary["roundIndex"].map(
        round_display_name
    )

    # Merge the compact summary into the detailed output so everything
    # needed for reporting is available in one CSV.
    output = summary.merge(
        ai_summary[
            [
                "awarenessCode",
                "roundIndex",
                "participantCount",
                "aiSupportedCount",
                "aiSupportedPercentage",
            ]
        ],
        on=["awarenessCode", "roundIndex"],
        how="left",
    )

    save_table(output, slug, index=False)

    # ------------------------------------------------------------------
    # Plot: specific workflow distribution by awareness
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(6.2 * len(groups), 5.5),
        sharey=True,
        squeeze=False,
    )

    for ax, group in zip(axes.flatten(), groups):
        group_summary = summary[
            summary["awarenessCode"].eq(group)
        ]
        bottoms = np.zeros(len(rounds), dtype=float)

        for workflow in WORKFLOW_ORDER:
            values = (
                group_summary[
                    group_summary["workflow"].eq(workflow)
                ]
                .set_index("roundIndex")
                .reindex(rounds)
            )

            percentages = values["choicePercentage"].to_numpy(
                dtype=float
            )
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

            for bar, percent, count, bottom in zip(
                    bars,
                    percentages,
                    counts,
                    bottoms,
            ):
                if percent >= 9:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + percent / 2,
                        f"{count}\n{percent:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=VALUE_LABEL_FONT_SIZE,
                        color="black",
                        )

            bottoms += np.nan_to_num(percentages)

        group_n = post.loc[
            post["awarenessCode"].eq(group),
            "participantId",
        ].nunique()

        ax.set_xticks(np.arange(len(rounds)))
        ax.set_xticklabels(
            [round_display_name(r) for r in rounds]
        )
        ax.set_ylim(0, 100)
        ax.set_xlabel("Post-error main round")
        ax.set_title(
            f"{AWARENESS_LABELS.get(group, group)} (n={group_n})"
        )

        apply_standard_axes_style(ax, grid_axis="y")

    axes[0, 0].set_ylabel("Workflow choices (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Workflow selected",
        bbox_to_anchor=(0.87, 0.5),
        loc="center left",
    )

    fig.suptitle(
        "Post-Error Workflow Choices by Error Awareness",
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0.045, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Error Awareness",
        "Distribution of voluntary workflow choices in Main Rounds 2-3 "
        "among participants exposed to the injected error, separated by "
        "whether they reported noticing the error.",
    )
