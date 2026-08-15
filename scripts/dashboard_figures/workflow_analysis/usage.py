"""Workflow usage, first choice, distribution, and participant trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from scripts.config import (
    AI_SUPPORTED_WORKFLOWS,
    MAIN_ROUND_INDICES,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    round_display_name,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    BAR_EDGE_COLOR,
    apply_standard_axes_style, VALUE_LABEL_FONT_SIZE, SUBTITLE_FONT_SIZE,
)
from scripts.utils import save_figure, save_table


def plot_total_workflow_usage_counts(main_df):
    """Plot pooled workflow selections across all main rounds."""
    slug = "01_total_workflow_usage_main_rounds"

    counts = (
        main_df["workflow"]
        .value_counts()
        .reindex(WORKFLOW_ORDER, fill_value=0)
        .astype(int)
    )
    total_selections = int(counts.sum())
    if total_selections == 0:
        return

    ai_count = int(counts.loc[AI_SUPPORTED_WORKFLOWS].sum())

    export_df = pd.DataFrame(
        {
            "workflow": [
                workflow_display_name(workflow) for workflow in WORKFLOW_ORDER
            ],
            "selection_count": counts.values,
            "percentage_of_all_main_round_selections": (counts / total_selections * 100)
            .round(2)
            .values,
        }
    )
    save_table(export_df, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER]
    bars = ax.bar(
        labels,
        counts.values,
        color=[WORKFLOW_COLORS[workflow] for workflow in WORKFLOW_ORDER],
        edgecolor=BAR_EDGE_COLOR,
    )
    apply_standard_axes_style(ax)

    for bar, count in zip(bars, counts.values):
        percentage = count / total_selections * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_selections * 0.015, 0.15),
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )

    ax.set_title("Total Workflow Usage in Main Rounds")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Workflow selections")
    ax.set_ylim(0, max(counts.max() * 1.25, 1))
    ax.tick_params(axis="x")
    ax.text(
        0.5,
        -0.22,
        (
            "AI-involving selections: "
            f"{ai_count}/{total_selections} "
            f"({ai_count / total_selections * 100:.1f}%)"
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=SUBTITLE_FONT_SIZE,
    )

    save_figure(
        fig,
        slug,
        "Total Workflow Usage in Main Rounds",
        "Pooled count and share of all workflow selections across the three "
        f"main rounds (N={total_selections} selections).",
    )


def plot_first_voluntary_workflow_choice(main_df):
    """Plot the choice made in the planned first main round."""
    slug = "02_first_voluntary_workflow_choice"

    main_rounds = sorted(main_df["roundIndex"].dropna().unique())
    if MAIN_ROUND_INDICES[0] not in main_rounds:
        return

    first_choice_df = main_df[main_df["roundIndex"] == MAIN_ROUND_INDICES[0]].copy()
    counts = (
        first_choice_df["workflow"]
        .value_counts()
        .reindex(WORKFLOW_ORDER, fill_value=0)
        .astype(int)
    )
    total = int(counts.sum())

    if total == 0:
        return

    ai_count = int(counts.loc[AI_SUPPORTED_WORKFLOWS].sum())

    export_df = pd.DataFrame(
        {
            "workflow": [
                workflow_display_name(workflow) for workflow in WORKFLOW_ORDER
            ],
            "count": counts.values,
            "percentage": (counts / total * 100).round(2).values,
        }
    )
    save_table(export_df, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER]
    bars = ax.bar(
        labels,
        counts.values,
        color=[WORKFLOW_COLORS[workflow] for workflow in WORKFLOW_ORDER],
        edgecolor=BAR_EDGE_COLOR,
    )
    apply_standard_axes_style(ax)

    for bar, count in zip(bars, counts.values):
        percentage = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.015, 0.15),
            f"{count} ({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONT_SIZE,
        )

    ax.set_title("First Voluntary Workflow Choice After Practice")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Participants")
    ax.set_ylim(0, max(counts.max() * 1.25, 1))
    ax.tick_params(axis="x")
    ax.text(
        0.5,
        -0.22,
        (f"AI-involving workflows: {ai_count}/{total} ({ai_count / total * 100:.1f}%)"),
        transform=ax.transAxes,
        ha="center",
        fontsize=SUBTITLE_FONT_SIZE,
    )

    save_figure(
        fig,
        slug,
        "First Voluntary Workflow Choice After Practice",
        "Distribution of workflow selections in Main 1, the first main "
        f"round after practice (N={total}).",
    )


def plot_workflow_distribution(main_df):
    """Plot workflow distributions separately for all available main rounds."""
    slug = "03_workflow_distribution_main_rounds"

    main_rounds = sorted(main_df["roundIndex"].dropna().unique())
    if len(main_rounds) < 2:
        return

    counts = (
        main_df.groupby(["roundIndex", "workflow"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=main_rounds, columns=WORKFLOW_ORDER, fill_value=0)
    )
    totals = counts.sum(axis=1)
    percentages = counts.div(totals.replace(0, pd.NA), axis=0).mul(100).fillna(0)
    ai_counts = counts.loc[:, AI_SUPPORTED_WORKFLOWS].sum(axis=1)
    ai_percentages = (ai_counts / totals.replace(0, pd.NA) * 100).round(2)

    export_df = percentages.rename(columns=WORKFLOW_LABELS).copy()
    export_df.insert(
        0,
        "Main round",
        [round_display_name(round_index) for round_index in main_rounds],
    )
    export_df["AI-involving count"] = ai_counts.values
    export_df["Total count"] = totals.values
    export_df["AI-involving percentage"] = ai_percentages.values
    save_table(export_df, slug, index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    positions = np.arange(len(main_rounds))
    bottom = np.zeros(len(main_rounds))

    for workflow in WORKFLOW_ORDER:
        values = percentages[workflow].values
        ax.bar(
            positions,
            values,
            bottom=bottom,
            label=workflow_display_name(workflow),
            color=WORKFLOW_COLORS[workflow],
            edgecolor=BAR_EDGE_COLOR,
        )
        bottom += values

    apply_standard_axes_style(ax)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{round_display_name(round_index)}\n(n={int(totals.loc[round_index])})"
            for round_index in main_rounds
        ]
    )
    ax.set_ylim(0, 113)
    ax.set_title("Workflow Distribution Across Main Rounds")
    ax.set_xlabel("Main round")
    ax.set_ylabel("Share of participants (%)")

    for position, round_index in enumerate(main_rounds):
        ax.text(
            position,
            103,
            f"AI-involving\n{ai_percentages.loc[round_index]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=SUBTITLE_FONT_SIZE,
        )

    ax.legend(
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Workflow Distribution Across Main Rounds",
        "Share of participants selecting each workflow in every "
        "main round. Workflow colours and order are fixed across figures.",
    )


def plot_participant_workflow_trajectories(main_df):
    """Show each participant's main-round sequence as a compact matrix."""
    slug = "04_participant_workflow_trajectories"

    main_rounds = sorted(main_df["roundIndex"].dropna().unique())
    if len(main_rounds) < 2:
        return

    sequence_matrix = main_df.pivot_table(
        index="participantId",
        columns="roundIndex",
        values="workflow",
        aggfunc="first",
    ).reindex(columns=main_rounds)

    if sequence_matrix.empty:
        return

    workflow_codes = {workflow: index for index, workflow in enumerate(WORKFLOW_ORDER)}
    missing_code = len(WORKFLOW_ORDER)

    sort_codes = sequence_matrix.apply(
        lambda column: column.map(workflow_codes).fillna(missing_code)
    )
    sequence_matrix = sequence_matrix.loc[
        sort_codes.sort_values(by=main_rounds, kind="stable").index
    ]

    participant_codes = [
        f"P{index:02d}" for index in range(1, len(sequence_matrix) + 1)
    ]
    export_df = sequence_matrix.copy()
    export_df.insert(0, "participantCode", participant_codes)
    export_df = export_df.rename(
        columns={
            round_index: round_display_name(round_index) for round_index in main_rounds
        }
    )
    save_table(export_df, slug, index=False)

    display_matrix = sequence_matrix.apply(
        lambda column: column.map(workflow_codes).fillna(missing_code)
    )

    cmap = ListedColormap(
        [WORKFLOW_COLORS[workflow] for workflow in WORKFLOW_ORDER] + ["#efefef"]
    )
    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, len(WORKFLOW_ORDER) + 1.5, 1),
        ncolors=cmap.N,
    )

    figure_height = max(
        4.0,
        min(10.5, 0.28 * len(sequence_matrix) + 2.1),
    )
    fig, ax = plt.subplots(figsize=(7.8, figure_height))
    ax.imshow(
        display_matrix.values,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    for row_index in range(display_matrix.shape[0] + 1):
        ax.axhline(row_index - 0.5, color="white", linewidth=0.8)
    for column_index in range(display_matrix.shape[1] + 1):
        ax.axvline(column_index - 0.5, color="white", linewidth=0.8)

    ax.set_xticks(range(len(main_rounds)))
    ax.set_xticklabels([round_display_name(round_index) for round_index in main_rounds])
    ax.set_ylabel("Participants (sorted by trajectory)")
    ax.set_title("Participant Workflow Trajectories Across Main Rounds")

    if len(participant_codes) <= 30:
        ax.set_yticks(range(len(participant_codes)))
        ax.set_yticklabels(participant_codes, fontsize=7)
    else:
        ax.set_yticks([])

    legend_handles = [
        patches.Patch(
            color=WORKFLOW_COLORS[workflow],
            label=workflow_display_name(workflow),
        )
        for workflow in WORKFLOW_ORDER
    ]
    if sequence_matrix.isna().any().any():
        legend_handles.append(patches.Patch(color="#efefef", label="Missing"))

    ax.legend(
        handles=legend_handles,
        title="Workflow",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Participant Workflow Trajectories Across Main Rounds",
        "Each row represents one anonymous participant and each column one main round.",
    )
