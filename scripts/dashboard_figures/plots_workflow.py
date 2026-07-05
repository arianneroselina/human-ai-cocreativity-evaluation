"""Workflow selection behaviour figures.

Core workflow figures
---------------------
01  Total workflow usage across all main rounds
02  First voluntary workflow choice after practice rounds
03  Workflow distribution across Main Rounds 1–3
04  Participant-level workflow trajectories across the main rounds
05a Final practice workflow -> first voluntary workflow choice
05b Main-round workflow transition heatmaps (Main 1 -> 2 and Main 2 -> 3)
06  Workflow retention after a voluntary choice
07  Number and pattern of workflow switches
08  Participant-reported workflow preference rank distribution
09  Stated workflow preference versus revealed workflow behaviour
"""

import json
import re

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from scripts.config import (
    AI_SUPPORTED_WORKFLOWS,
    MAIN_ROUND_INDICES,
    PRACTICE_ROUND_INDICES,
    WORKFLOW_COLORS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    workflow_display_name,
    round_display_name,
    phase_data,
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


RANK_COLORS = ["C0", "C1", "C2", "C3"]


# -----------------------------------------------------------------------------
# Shared workflow helpers
# -----------------------------------------------------------------------------


def _complete_main_sequences(main_df):
    """Return participants with an observation in every planned main round."""
    if main_df.empty:
        return pd.DataFrame(columns=MAIN_ROUND_INDICES)

    sequence_matrix = main_df.pivot_table(
        index="participantId",
        columns="roundIndex",
        values="workflow",
        aggfunc="first",
    ).reindex(columns=MAIN_ROUND_INDICES)

    return sequence_matrix.dropna(how="any")


def _transition_rows_for_pair(round_df, from_round, to_round):
    """Create transitions for one explicit consecutive round pair only."""
    source = round_df.loc[
        round_df["roundIndex"] == from_round,
        ["participantId", "workflow"],
    ].rename(columns={"workflow": "fromWorkflow"})

    target = round_df.loc[
        round_df["roundIndex"] == to_round,
        ["participantId", "workflow"],
    ].rename(columns={"workflow": "toWorkflow"})

    transitions = source.merge(target, on="participantId", how="inner")
    transitions["fromRound"] = from_round
    transitions["toRound"] = to_round
    transitions["switched"] = transitions["fromWorkflow"] != transitions["toWorkflow"]

    return transitions


def _transition_matrix(transitions):
    """Return count and source-row-percentage matrices for transitions."""
    counts = (
        transitions.groupby(["fromWorkflow", "toWorkflow"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=WORKFLOW_ORDER, columns=WORKFLOW_ORDER, fill_value=0)
    )

    row_totals = counts.sum(axis=1).astype(float)
    row_percentages = (
        counts.div(row_totals.replace(0, np.nan), axis=0).mul(100).fillna(0.0)
    )

    return counts, row_percentages, row_totals.astype(int)


def _save_transition_heatmap(
    transitions,
    slug,
    title,
    source_axis_label,
    target_axis_label,
    description,
):
    """Save a transition heatmap with counts and source-row percentages."""
    if transitions.empty:
        return

    counts, row_percentages, row_totals = _transition_matrix(transitions)

    display_counts = counts.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS)
    display_percentages = row_percentages.rename(
        index=WORKFLOW_LABELS,
        columns=WORKFLOW_LABELS,
    )

    save_table(display_counts, f"{slug}_counts")
    save_table(display_percentages.round(2), f"{slug}_row_percentages")

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(
        row_percentages.values,
        vmin=0,
        vmax=100,
        cmap="Blues",
    )

    x_labels = [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER]
    y_labels = [
        (f"{workflow_display_name(workflow)}\n(n={int(row_totals.loc[workflow])})")
        if row_totals.loc[workflow] > 0
        else workflow_display_name(workflow)
        for workflow in WORKFLOW_ORDER
    ]

    ax.set_title(title)
    ax.set_xlabel(target_axis_label)
    ax.set_ylabel(source_axis_label)
    ax.set_xticks(range(len(WORKFLOW_ORDER)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(WORKFLOW_ORDER)))
    ax.set_yticklabels(y_labels)

    for row_index, from_workflow in enumerate(WORKFLOW_ORDER):
        for column_index, to_workflow in enumerate(WORKFLOW_ORDER):
            count = int(counts.loc[from_workflow, to_workflow])
            percentage = float(row_percentages.loc[from_workflow, to_workflow])

            if row_totals.loc[from_workflow] == 0:
                label = "–"
            else:
                label = f"{count}\n{percentage:.0f}%"

            text_color = "white" if percentage >= 55 else "black"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Share within source workflow (%)",
    )

    save_figure(fig, slug, title, description)


# -----------------------------------------------------------------------------
# 01–03: Aggregate workflow choices
# -----------------------------------------------------------------------------


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
            fontsize=9,
        )

    ax.set_title("Total Workflow Usage in Main Rounds")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Workflow selections")
    ax.set_ylim(0, max(counts.max() * 1.25, 1))
    ax.tick_params(axis="x", rotation=15)
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
        fontsize=9,
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
    if len(main_rounds) < 2:
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
            fontsize=9,
        )

    ax.set_title("First Voluntary Workflow Choice After Practice")
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Participants")
    ax.set_ylim(0, max(counts.max() * 1.25, 1))
    ax.tick_params(axis="x", rotation=15)
    ax.text(
        0.5,
        -0.22,
        (f"AI-involving workflows: {ai_count}/{total} ({ai_count / total * 100:.1f}%)"),
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    save_figure(
        fig,
        slug,
        "First Voluntary Workflow Choice After Practice",
        "Distribution of workflow selections in Main 1, the first main"
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
    ax.set_xlabel("Free-choice round")
    ax.set_ylabel("Share of participants (%)")

    for position, round_index in enumerate(main_rounds):
        ax.text(
            position,
            103,
            f"AI-involving\n{ai_percentages.loc[round_index]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
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


# -----------------------------------------------------------------------------
# 04–05: Individual trajectories and transitions
# -----------------------------------------------------------------------------


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


def plot_practice_to_first_choice_transition(prepared: pd.DataFrame) -> None:
    """Compare the final assigned practice workflow with the first choice."""
    slug = "05_final_practice_to_first_voluntary_choice"

    if prepared.empty:
        return

    final_practice_round = PRACTICE_ROUND_INDICES[-1]
    first_main_round = MAIN_ROUND_INDICES[0]

    transitions = _transition_rows_for_pair(
        prepared,
        from_round=final_practice_round,
        to_round=first_main_round,
    )

    if transitions.empty:
        return

    counts, percentages, totals = _transition_matrix(transitions)

    save_table(
        counts.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS),
        f"{slug}_counts",
    )
    save_table(
        percentages.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS).round(2),
        f"{slug}_row_percentages",
    )

    from_label = round_display_name(final_practice_round)
    to_label = round_display_name(first_main_round)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    image = ax.imshow(
        percentages.values,
        vmin=0,
        vmax=100,
        cmap="Blues",
    )

    ax.set_title("Final Practice Workflow to First Voluntary Choice")
    ax.set_xlabel(f"Chosen workflow in {to_label}")
    ax.set_ylabel(f"Assigned workflow in {from_label}")

    ax.set_xticks(range(len(WORKFLOW_ORDER)))
    ax.set_xticklabels(
        [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(len(WORKFLOW_ORDER)))
    ax.set_yticklabels(
        [
            (f"{workflow_display_name(workflow)}\n(n={int(totals.loc[workflow])})")
            if totals.loc[workflow] > 0
            else workflow_display_name(workflow)
            for workflow in WORKFLOW_ORDER
        ]
    )

    for row, source_workflow in enumerate(WORKFLOW_ORDER):
        for col, target_workflow in enumerate(WORKFLOW_ORDER):
            total = totals.loc[source_workflow]
            count = counts.loc[source_workflow, target_workflow]
            percentage = percentages.loc[source_workflow, target_workflow]

            ax.text(
                col,
                row,
                "–" if total == 0 else f"{int(count)}\n{percentage:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if percentage >= 55 else "black",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Share within assigned practice workflow (%)",
    )

    save_figure(
        fig,
        slug,
        "Final Practice Workflow to First Voluntary Choice",
        "Transition from the final assigned practice round to the first voluntary "
        "workflow choice. Cells show participant counts and row percentages within "
        "the assigned final-practice workflow.",
    )


def plot_main_workflow_transitions(main_df) -> None:
    """Show all consecutive Main-round workflow transitions in one figure."""
    slug = "05b_main_workflow_transitions"

    main_rounds = sorted(main_df["roundIndex"].dropna().unique())
    if len(main_rounds) < 2:
        return

    transition_data = []

    for from_round, to_round in zip(main_rounds, main_rounds[1:]):
        transitions = _transition_rows_for_pair(
            main_df,
            from_round,
            to_round,
        )

        if transitions.empty:
            continue

        counts, percentages, totals = _transition_matrix(transitions)

        from_label = round_display_name(from_round)
        to_label = round_display_name(to_round)
        table_slug = (
            f"05b_transition_{from_label.lower().replace(' ', '_')}_to_"
            f"{to_label.lower().replace(' ', '_')}"
        )

        save_table(
            counts.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS),
            f"{table_slug}_counts",
        )
        save_table(
            percentages.rename(index=WORKFLOW_LABELS, columns=WORKFLOW_LABELS).round(2),
            f"{table_slug}_row_percentages",
        )

        transition_data.append((from_label, to_label, counts, percentages, totals))

    if not transition_data:
        return

    fig, axes = plt.subplots(
        1,
        len(transition_data),
        figsize=(7.2 * len(transition_data), 5.8),
        sharey=False,  # Important: n differs per transition
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for index, (ax, data) in enumerate(zip(axes, transition_data)):
        from_label, to_label, counts, percentages, totals = data

        image = ax.imshow(
            percentages.values,
            vmin=0,
            vmax=100,
            cmap="Blues",
        )

        ax.set_title(f"{from_label} → {to_label}")
        ax.set_xlabel(f"Workflow in {to_label}")
        ax.set_xticks(range(len(WORKFLOW_ORDER)))
        ax.set_xticklabels(
            [workflow_display_name(w) for w in WORKFLOW_ORDER],
            rotation=30,
            ha="right",
        )

        ax.set_yticks(range(len(WORKFLOW_ORDER)))
        ax.set_yticklabels(
            [
                (f"{workflow_display_name(w)}\n(n={int(totals.loc[w])})")
                if totals.loc[w] > 0
                else workflow_display_name(w)
                for w in WORKFLOW_ORDER
            ]
        )
        ax.tick_params(axis="y", labelleft=True)

        if index == 0:
            ax.set_ylabel(f"Workflow in {from_label}")

        for row, source_workflow in enumerate(WORKFLOW_ORDER):
            for col, target_workflow in enumerate(WORKFLOW_ORDER):
                total = totals.loc[source_workflow]
                count = counts.loc[source_workflow, target_workflow]
                percentage = percentages.loc[source_workflow, target_workflow]

                label = "–" if total == 0 else f"{int(count)}\n{percentage:.0f}%"

                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if percentage >= 55 else "black",
                )

    fig.colorbar(
        image,
        ax=axes.tolist(),
        label="Share within source workflow (%)",
        pad=0.03,
    )

    fig.suptitle(
        "Workflow Transitions Across Main Rounds",
        y=0.985,
    )

    save_figure(
        fig,
        slug,
        "Workflow Transitions Across Main Rounds",
        "Each cell shows the participant count and the percentage within the workflow selected in the preceding Main round.",
    )


# -----------------------------------------------------------------------------
# 06–07: Retention and switching behaviour
# -----------------------------------------------------------------------------


def plot_workflow_retention(main_df):
    """Show how often each voluntary workflow is retained in the next round."""
    slug = "06_workflow_retention"

    main_rounds = sorted(main_df["roundIndex"].dropna().unique())
    if len(main_rounds) < 2:
        return

    transition_sets = [
        _transition_rows_for_pair(main_df, from_round, to_round)
        for from_round, to_round in zip(main_rounds, main_rounds[1:])
    ]
    transitions = pd.concat(transition_sets, ignore_index=True)

    if transitions.empty:
        return

    summary = (
        transitions.groupby("fromWorkflow")
        .agg(
            eligible_transitions=("participantId", "size"),
            retained=("switched", lambda values: int((~values).sum())),
            switched=("switched", "sum"),
        )
        .reindex(WORKFLOW_ORDER, fill_value=0)
    )
    summary["retention_percentage"] = np.where(
        summary["eligible_transitions"] > 0,
        summary["retained"] / summary["eligible_transitions"] * 100,
        np.nan,
    )
    summary.insert(
        0,
        "workflow",
        [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER],
    )
    save_table(summary, slug, index=False)

    transition_labels = {
        (from_round, to_round): (
            f"{round_display_name(from_round)} → {round_display_name(to_round)}"
        )
        for from_round, to_round in zip(main_rounds, main_rounds[1:])
    }
    by_step = transitions.copy()
    by_step["transition"] = [
        transition_labels[(row.fromRound, row.toRound)]
        for row in by_step.itertuples(index=False)
    ]
    by_step = (
        by_step.groupby(["transition", "fromWorkflow"])
        .agg(
            eligible_transitions=("participantId", "size"),
            retained=("switched", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    by_step["retention_percentage"] = np.where(
        by_step["eligible_transitions"] > 0,
        by_step["retained"] / by_step["eligible_transitions"] * 100,
        np.nan,
    )
    by_step["fromWorkflow"] = by_step["fromWorkflow"].map(workflow_display_name)
    save_table(by_step, f"{slug}_by_transition", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER]
    values = summary["retention_percentage"].fillna(0).values
    bars = ax.bar(
        labels,
        values,
        color=[WORKFLOW_COLORS[workflow] for workflow in WORKFLOW_ORDER],
        edgecolor=BAR_EDGE_COLOR,
    )
    apply_standard_axes_style(ax)

    for bar, (_, row) in zip(bars, summary.iterrows()):
        eligible = int(row["eligible_transitions"])
        retained = int(row["retained"])
        label = (
            "No next-round\nobservations" if eligible == 0 else f"{retained}/{eligible}"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, 112)
    ax.set_title("Workflow Retention in the Next Main Round")
    ax.set_xlabel("Workflow selected in the earlier round")
    ax.set_ylabel("Retained in next round (%)")
    ax.tick_params(axis="x", rotation=15)

    save_figure(
        fig,
        slug,
        "Workflow Retention in the Next Main Round",
        "Probability of choosing the same workflow in the immediately following "
        "main round.",
    )


def _classify_switch_pattern(sequence):
    """Classify a complete three-round workflow sequence."""
    first, second, third = sequence

    if first == second == third:
        return "Stayed with one workflow"
    if first == third and first != second:
        return "Switched back"
    if first == second != third or first != second == third:
        return "Switched and stayed"
    if len({first, second, third}) == 3:
        return "Explored three workflows"

    return "Other switching pattern"


def plot_workflow_switching_behavior(main_df):
    """Plot the number and pattern of switches across all main rounds."""
    slug = "07_workflow_switching_behavior"

    sequences = _complete_main_sequences(main_df)
    if sequences.empty:
        return

    analysis_df = sequences.copy()
    analysis_df["switchCount"] = analysis_df.apply(
        lambda row: sum(
            row.iloc[index] != row.iloc[index + 1]
            for index in range(len(MAIN_ROUND_INDICES) - 1)
        ),
        axis=1,
    )
    analysis_df["switchCategory"] = analysis_df["switchCount"].map(
        {
            0: "0 switches",
            1: "1 switch",
            2: "2 switches",
        }
    )
    analysis_df["switchPattern"] = analysis_df.apply(
        lambda row: _classify_switch_pattern(row.loc[MAIN_ROUND_INDICES].tolist()),
        axis=1,
    )

    participant_codes = [f"P{index:02d}" for index in range(1, len(analysis_df) + 1)]
    sequence_export = analysis_df.reset_index().copy()
    sequence_export.insert(0, "participantCode", participant_codes)
    sequence_export = sequence_export.drop(columns="participantId")
    sequence_export = sequence_export.rename(
        columns={
            round_index: round_display_name(round_index)
            for round_index in MAIN_ROUND_INDICES
        }
    )
    save_table(
        sequence_export,
        f"{slug}_participant_sequences",
        index=False,
    )

    total = len(analysis_df)
    switch_order = ["0 switches", "1 switch", "2 switches"]
    switch_counts = (
        analysis_df["switchCategory"]
        .value_counts()
        .reindex(
            switch_order,
            fill_value=0,
        )
    )
    pattern_order = [
        "Stayed with one workflow",
        "Switched and stayed",
        "Switched back",
        "Explored three workflows",
        "Other switching pattern",
    ]
    pattern_counts = (
        analysis_df["switchPattern"]
        .value_counts()
        .reindex(
            pattern_order,
            fill_value=0,
        )
    )

    save_table(
        pd.DataFrame(
            {
                "category": switch_counts.index,
                "count": switch_counts.values,
                "percentage": (switch_counts / total * 100).round(2).values,
            }
        ),
        f"{slug}_switch_counts",
        index=False,
    )
    save_table(
        pd.DataFrame(
            {
                "pattern": pattern_counts.index,
                "count": pattern_counts.values,
                "percentage": (pattern_counts / total * 100).round(2).values,
            }
        ),
        f"{slug}_patterns",
        index=False,
    )

    fig, (ax_count, ax_pattern) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.0),
        layout="constrained",
    )

    count_bars = ax_count.bar(
        switch_counts.index,
        switch_counts.values,
        edgecolor=BAR_EDGE_COLOR,
    )
    apply_standard_axes_style(ax_count)
    ax_count.set_title("Number of Workflow Switches")
    ax_count.set_xlabel("Switches across all main rounds")
    ax_count.set_ylabel("Participants")
    ax_count.set_ylim(0, max(switch_counts.max() * 1.3, 1))

    for bar, count in zip(count_bars, switch_counts.values):
        ax_count.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total * 0.02, 0.12),
            f"{count} ({count / total * 100:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    pattern_bars = ax_pattern.barh(
        pattern_counts.index,
        pattern_counts.values,
        edgecolor=BAR_EDGE_COLOR,
    )
    apply_standard_axes_style(ax_pattern, grid_axis="x")
    ax_pattern.set_title("Switching Patterns")
    ax_pattern.set_xlabel("Participants")
    ax_pattern.set_xlim(0, max(pattern_counts.max() * 1.35, 1))

    for bar, count in zip(pattern_bars, pattern_counts.values):
        ax_pattern.text(
            bar.get_width() + max(total * 0.015, 0.1),
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({count / total * 100:.0f}%)",
            va="center",
            fontsize=9,
        )

    save_figure(
        fig,
        slug,
        "Workflow Switching Behaviour Across Main Rounds",
        "The left panel shows the number of switches; the right panel distinguishes "
        "stable use, switching and staying, switching back, and exploration.",
    )


# -----------------------------------------------------------------------------
# 08–09: Stated preferences versus revealed behaviour
# -----------------------------------------------------------------------------


def normalize_ranking(items):
    """Normalise ranking entries to configured workflow identifiers."""
    result = []

    for item in items:
        normalized = str(item).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace("→", "_")
        normalized = re.sub(r"_+", "_", normalized)

        if normalized in WORKFLOW_ORDER:
            result.append(normalized)

    return result


def parse_workflow_ranking(value):
    """Parse JSON and legacy string representations of a workflow ranking."""
    if pd.isna(value):
        return []

    raw = str(value).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ranking = []

            for item in parsed:
                if isinstance(item, str):
                    ranking.append(item)
                elif isinstance(item, dict) and "workflow" in item:
                    ranking.append(str(item["workflow"]))

            return normalize_ranking(ranking)
    except json.JSONDecodeError:
        pass

    raw = raw.strip("{}[]()")
    return normalize_ranking(re.split(r"[>,;|\n]+", raw))


def _build_valid_ranking_rows(feedback_df):
    """Return complete rankings plus an audit table for invalid entries."""
    required = {"sessionId", "workflowRanking"}

    if feedback_df.empty or not require_columns(
        feedback_df,
        required,
        "workflow preference ranking",
    ):
        return pd.DataFrame(), pd.DataFrame()

    feedback = feedback_df.drop_duplicates(
        "sessionId",
        keep="last",
    ).copy()
    ranking_rows = []
    audit_rows = []
    expected_workflows = set(WORKFLOW_ORDER)

    for _, row in feedback.iterrows():
        ranking = parse_workflow_ranking(row["workflowRanking"])
        is_valid = (
            len(ranking) == len(WORKFLOW_ORDER) and set(ranking) == expected_workflows
        )

        audit_rows.append(
            {
                "sessionId": row["sessionId"],
                "validRanking": is_valid,
                "parsedWorkflowCount": len(ranking),
                "ranking": " > ".join(ranking),
            }
        )

        if not is_valid:
            continue

        for rank, workflow in enumerate(ranking, start=1):
            ranking_rows.append(
                {
                    "sessionId": row["sessionId"],
                    "workflow": workflow,
                    "rank": rank,
                }
            )

    return pd.DataFrame(ranking_rows), pd.DataFrame(audit_rows)


def _ranking_summary(ranking_rows):
    """Build rank counts and mean rank for complete valid rankings."""
    rank_counts = (
        ranking_rows.groupby(["workflow", "rank"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=WORKFLOW_ORDER,
            columns=range(1, len(WORKFLOW_ORDER) + 1),
            fill_value=0,
        )
    )
    rank_counts["meanRank"] = (
        ranking_rows.groupby("workflow")["rank"].mean().reindex(WORKFLOW_ORDER)
    )

    return rank_counts


def plot_workflow_preference_ranking(
    ranking_rows: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    """Visualise rank distributions and mean stated preference."""
    slug = "08_workflow_preference_rank_distribution"

    if not audit_df.empty:
        save_table(audit_df, f"{slug}_ranking_audit", index=False)

    if ranking_rows.empty:
        return

    summary = _ranking_summary(ranking_rows)
    workflow_order = summary["meanRank"].sort_values().index.tolist()
    display_summary = summary.loc[workflow_order].copy()

    export_df = display_summary.rename(index=WORKFLOW_LABELS).rename(
        columns={rank: f"Rank {rank}" for rank in range(1, len(WORKFLOW_ORDER) + 1)}
    )
    save_table(export_df, slug)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    positions = np.arange(len(workflow_order))
    left = np.zeros(len(workflow_order))
    valid_participants = ranking_rows["sessionId"].nunique()

    for rank in range(1, len(WORKFLOW_ORDER) + 1):
        values = display_summary[rank].values
        ax.barh(
            positions,
            values,
            left=left,
            label=f"Rank {rank}",
            color=RANK_COLORS[rank - 1],
            edgecolor=BAR_EDGE_COLOR,
        )
        left += values

    apply_standard_axes_style(ax, grid_axis="x")

    for index, workflow in enumerate(workflow_order):
        mean_rank = display_summary.loc[workflow, "meanRank"]
        ax.text(
            valid_participants + max(valid_participants * 0.02, 0.15),
            index,
            f"Mean rank: {mean_rank:.2f}",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([workflow_display_name(workflow) for workflow in workflow_order])
    ax.invert_yaxis()
    ax.set_xlim(0, valid_participants * 1.24)
    ax.set_title("Participant-Reported Workflow Preference Ranking")
    ax.set_xlabel("Participants")
    ax.legend(
        title="Assigned rank",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Participant-Reported Workflow Preference Ranking",
        "Distribution of first through fourth preference rankings (N={valid_participants}). Lower mean rank "
        "indicates stronger stated preference.",
    )


def _plot_choice_crosstab(ax, matrix, title, row_label, column_label):
    """Draw a workflow-by-workflow crosstab heatmap."""
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel(column_label)
    ax.set_ylabel(row_label)
    ax.set_xticks(range(len(WORKFLOW_ORDER)))
    ax.set_xticklabels(
        [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(WORKFLOW_ORDER)))
    ax.set_yticklabels([workflow_display_name(workflow) for workflow in WORKFLOW_ORDER])

    maximum = matrix.values.max()

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix.iloc[row_index, column_index])
            text_color = "white" if maximum > 0 and value > maximum / 2 else "black"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    return image


def plot_stated_vs_revealed_workflow_behavior(
    ranking_rows: pd.DataFrame,
    main_df: pd.DataFrame,
) -> None:
    """Compare stated top preference with first and modal actual choices."""
    slug = "09_stated_vs_revealed_workflow_behavior"

    if ranking_rows.empty:
        return

    stated_top = ranking_rows[ranking_rows["rank"] == 1].rename(
        columns={"workflow": "statedTop"}
    )[["sessionId", "statedTop"]]
    first_choice = (
        main_df[main_df["roundIndex"] == MAIN_ROUND_INDICES[0]][
            ["sessionId", "workflow"]
        ]
        .drop_duplicates("sessionId", keep="last")
        .rename(columns={"workflow": "firstChoice"})
    )

    first_joined = stated_top.merge(
        first_choice,
        on="sessionId",
        how="inner",
    )
    if first_joined.empty:
        return

    first_matrix = pd.crosstab(
        first_joined["statedTop"], first_joined["firstChoice"]
    ).reindex(
        index=WORKFLOW_ORDER,
        columns=WORKFLOW_ORDER,
        fill_value=0,
    )
    save_table(first_matrix, f"{slug}_first_choice")

    first_agreement = int(
        (first_joined["statedTop"] == first_joined["firstChoice"]).sum()
    )
    first_n = len(first_joined)

    sequences = _complete_main_sequences(main_df)
    session_by_participant = (
        main_df[["participantId", "sessionId"]]
        .dropna()
        .drop_duplicates("participantId", keep="last")
        .set_index("participantId")["sessionId"]
    )

    modal_rows = []
    for participant_id, row in sequences.iterrows():
        modes = row.value_counts()
        top_workflows = modes[modes == modes.max()].index.tolist()
        modal_rows.append(
            {
                "participantId": participant_id,
                "modalChoice": (top_workflows[0] if len(top_workflows) == 1 else pd.NA),
                "modalTie": len(top_workflows) > 1,
            }
        )

    modal_choice = pd.DataFrame(modal_rows)
    if not modal_choice.empty:
        modal_choice["sessionId"] = modal_choice["participantId"].map(
            session_by_participant
        )
        modal_choice = modal_choice.dropna(subset=["sessionId"])

    if modal_choice.empty:
        modal_joined = pd.DataFrame()
        modal_matrix = pd.DataFrame(
            0,
            index=WORKFLOW_ORDER,
            columns=WORKFLOW_ORDER,
        )
        modal_agreement = 0
        modal_n = 0
        modal_ties = 0
    else:
        modal_joined = stated_top.merge(
            modal_choice.dropna(subset=["modalChoice"])[["sessionId", "modalChoice"]],
            on="sessionId",
            how="inner",
        )
        modal_matrix = pd.crosstab(
            modal_joined["statedTop"],
            modal_joined["modalChoice"],
        ).reindex(
            index=WORKFLOW_ORDER,
            columns=WORKFLOW_ORDER,
            fill_value=0,
        )
        modal_agreement = int(
            (modal_joined["statedTop"] == modal_joined["modalChoice"]).sum()
        )
        modal_n = len(modal_joined)
        modal_ties = int(modal_choice["modalTie"].sum())

    save_table(modal_matrix, f"{slug}_modal_choice")

    agreement_summary = pd.DataFrame(
        [
            {
                "comparison": "Stated Rank 1 vs first voluntary choice",
                "pairedParticipants": first_n,
                "agreementCount": first_agreement,
                "agreementPercentage": round(
                    first_agreement / first_n * 100,
                    2,
                ),
                "excludedModalTies": 0,
            },
            {
                "comparison": ("Stated Rank 1 vs unique modal main-round choice"),
                "pairedParticipants": modal_n,
                "agreementCount": modal_agreement,
                "agreementPercentage": (
                    round(modal_agreement / modal_n * 100, 2) if modal_n else np.nan
                ),
                "excludedModalTies": modal_ties,
            },
        ]
    )
    save_table(
        agreement_summary,
        f"{slug}_agreement_summary",
        index=False,
    )

    fig, (first_ax, modal_ax) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        layout="constrained",
    )

    first_image = _plot_choice_crosstab(
        first_ax,
        first_matrix,
        (
            f"First voluntary choice\nAgreement: "
            f"{first_agreement}/{first_n} "
            f"({first_agreement / first_n * 100:.0f}%)"
        ),
        "Stated Rank 1 preference",
        "First voluntary choice",
    )

    modal_title = (
        "Most-used workflow across main rounds\n"
        f"Agreement: {modal_agreement}/{modal_n} "
        f"({modal_agreement / modal_n * 100:.0f}%)"
        if modal_n
        else ("Most-used workflow across main rounds\nNo unique modal choices")
    )
    _plot_choice_crosstab(
        modal_ax,
        modal_matrix,
        modal_title,
        "Stated Rank 1 preference",
        "Unique modal workflow",
    )

    fig.colorbar(
        first_image,
        ax=[first_ax, modal_ax],
        label="Participants",
    )

    save_figure(
        fig,
        slug,
        "Stated Workflow Preference Versus Revealed Workflow Behaviour",
        "Rows show the participant's stated top-ranked workflow and columns show "
        "their actual first or uniquely most-used main-round workflow. Participants "
        "with a tie for most-used workflow are excluded only from the modal-choice "
        "panel.",
    )


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------


def plot_workflow(df, feedback_df):
    """Generate the workflow-selection figures in analysis order."""
    main_df = phase_data(df, "main")
    if main_df.empty:
        return

    ranking_rows, audit_df = _build_valid_ranking_rows(feedback_df)

    plot_total_workflow_usage_counts(main_df)
    plot_first_voluntary_workflow_choice(main_df)
    plot_workflow_distribution(main_df)
    plot_participant_workflow_trajectories(main_df)
    plot_practice_to_first_choice_transition(df)
    plot_main_workflow_transitions(main_df)
    plot_workflow_retention(main_df)
    plot_workflow_switching_behavior(main_df)
    plot_workflow_preference_ranking(ranking_rows, audit_df)
    plot_stated_vs_revealed_workflow_behavior(ranking_rows, main_df)
