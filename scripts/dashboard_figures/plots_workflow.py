"""Workflow selection behaviour figures.

Core workflow figures
---------------------
01  Total workflow usage across Main Rounds
02  First voluntary workflow choice after Practice Rounds
03  Workflow distribution across Main Rounds
04  Participant-level workflow trajectories across Main Rounds
05a Final practice workflow -> first voluntary workflow choice
05b Main-round workflow transition heatmaps (Main 1 -> 2 and Main 2 -> 3)
06  Workflow retention across Main Rounds
07  Number and pattern of workflow switches across Main Rounds
08  Final workflow preference rank average and distribution
09  Final workflow preference versus observed workflow behaviour
"""

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
    ranking_summary,
    build_valid_ranking_rows,
)
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
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


def _plot_transition_heatmap(
    ax,
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    totals: pd.Series,
    *,
    title: str,
    source_label: str,
    target_label: str,
):
    """Draw one transition matrix with counts and row percentages."""
    image = ax.imshow(
        percentages.to_numpy(),
        vmin=0,
        vmax=100,
        cmap="Blues",
    )

    ax.set_title(title)
    ax.set_xlabel(target_label)
    ax.set_ylabel(source_label)

    ax.set_xticks(range(len(WORKFLOW_ORDER)))
    ax.set_xticklabels(
        [workflow_display_name(w) for w in WORKFLOW_ORDER],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(range(len(WORKFLOW_ORDER)))
    ax.set_yticklabels(
        [
            (
                f"{workflow_display_name(w)}\n(n={int(totals.loc[w])})"
                if totals.loc[w] > 0
                else workflow_display_name(w)
            )
            for w in WORKFLOW_ORDER
        ]
    )

    for row, source in enumerate(WORKFLOW_ORDER):
        for column, target in enumerate(WORKFLOW_ORDER):
            total = int(totals.loc[source])
            count = int(counts.loc[source, target])
            percentage = float(percentages.loc[source, target])

            ax.text(
                column,
                row,
                "–" if total == 0 else f"{count}\n{percentage:.0f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if percentage >= 55 else "black",
            )

    return image


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

    image = _plot_transition_heatmap(
        ax,
        counts,
        percentages,
        totals,
        title="Final Practice Workflow to First Voluntary Choice",
        source_label=f"Assigned workflow in {from_label}",
        target_label=f"Chosen workflow in {to_label}",
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

        image = _plot_transition_heatmap(
            ax,
            counts,
            percentages,
            totals,
            title=f"{from_label} → {to_label}",
            source_label=f"Workflow in {from_label}",
            target_label=f"Workflow in {to_label}",
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


def plot_workflow_switching_behaviour(main_df):
    """Plot the number and pattern of switches across all main rounds."""
    slug = "07_workflow_switching_behaviour"

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
# 08: Stated preferences and grouped by reported AI errors
# -----------------------------------------------------------------------------


def plot_final_workflow_preference(
    ranking_rows: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> None:
    """Visualise average stated rank and the full rank distribution."""
    slug = "08_final_workflow_preference"

    if not audit_df.empty:
        save_table(audit_df, f"{slug}_ranking_audit", index=False)

    if ranking_rows.empty:
        return

    summary = ranking_summary(ranking_rows)
    rank_columns = list(range(1, len(WORKFLOW_ORDER) + 1))

    # Lower mean rank indicates stronger preference.
    workflow_order = summary["meanRank"].sort_values().index.tolist()
    display_summary = summary.loc[workflow_order].copy()

    valid_participants = ranking_rows["sessionId"].nunique()

    # Export counts and mean ranks.
    export_df = display_summary.rename(index=WORKFLOW_LABELS).rename(
        columns={rank: f"Rank {rank}" for rank in rank_columns}
    )
    save_table(export_df, slug)

    # Convert rank counts to percentages for the distribution panel.
    rank_counts = display_summary[rank_columns]
    row_totals = rank_counts.sum(axis=1)
    rank_percentages = rank_counts.div(row_totals, axis=0) * 100

    positions = np.arange(len(workflow_order))

    fig, (ax_mean, ax_distribution) = plt.subplots(
        ncols=2,
        figsize=(11.8, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.8]},
    )

    # ------------------------------------------------------------
    # Left panel: average rank
    # ------------------------------------------------------------
    mean_ranks = display_summary["meanRank"].to_numpy()

    ax_mean.hlines(
        y=positions,
        xmin=1,
        xmax=mean_ranks,
        color="0.75",
        linewidth=2,
        zorder=1,
    )

    for index, workflow in enumerate(workflow_order):
        mean_rank = display_summary.loc[workflow, "meanRank"]
        first_choice_count = int(display_summary.loc[workflow, 1])

        ax_mean.scatter(
            mean_rank,
            index,
            s=95,
            color=WORKFLOW_COLORS[workflow],
            edgecolor=BAR_EDGE_COLOR,
            zorder=2,
        )

        ax_mean.text(
            mean_rank + 0.08,
            index,
            f"{mean_rank:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        ax_mean.text(
            mean_rank + 0.08,
            index + 0.16,
            f"{first_choice_count} first-choice votes",
            va="center",
            fontsize=8,
            color="0.35",
        )

    ax_mean.set_yticks(positions)
    ax_mean.set_yticklabels(
        [
            f"{position + 1}. {workflow_display_name(workflow)}"
            for position, workflow in enumerate(workflow_order)
        ]
    )
    ax_mean.invert_yaxis()

    ax_mean.set_xlim(0.8, 4.35)
    ax_mean.set_xticks([1, 2, 3, 4])
    ax_mean.set_xticklabels(["1\nBest", "2", "3", "4\nWorst"])
    ax_mean.set_xlabel("Average assigned rank")
    ax_mean.set_title("Average preference rank")

    apply_standard_axes_style(ax_mean, grid_axis="x")

    # ------------------------------------------------------------
    # Right panel: complete rank distribution
    # ------------------------------------------------------------
    left = np.zeros(len(workflow_order))

    for rank in rank_columns:
        percentages = rank_percentages[rank].to_numpy()

        bars = ax_distribution.barh(
            positions,
            percentages,
            left=left,
            label=f"Rank {rank}",
            color=RANK_COLORS[rank - 1],
            edgecolor=BAR_EDGE_COLOR,
        )

        # Add percentages only where the segment is wide enough.
        for index, (bar, percentage) in enumerate(zip(bars, percentages)):
            if percentage >= 8:
                ax_distribution.text(
                    left[index] + percentage / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        left += percentages

    ax_distribution.set_xlim(0, 100)
    ax_distribution.set_xlabel("Participants assigning each rank (%)")
    ax_distribution.set_title("Distribution of assigned ranks")

    apply_standard_axes_style(ax_distribution, grid_axis="x")

    ax_distribution.legend(
        title="Assigned rank",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.suptitle(
        f"Final Workflow Preference (N={valid_participants})",
        fontsize=14,
    )

    fig.tight_layout()

    save_figure(
        fig,
        slug,
        "Final Workflow Preference",
        (
            f"Average and distribution of final workflow rankings from "
            f"{valid_participants} participants. Rank 1 represents the strongest "
            "preference. Each participant ranked every workflow once; rankings "
            "were not weighted by the frequency of workflow use in the main rounds."
        ),
    )


# -----------------------------------------------------------------------------
# 09: Stated preferences versus revealed behaviour
# -----------------------------------------------------------------------------


def _plot_row_percentage_crosstab(
    ax,
    counts: pd.DataFrame,
    title: str,
    ylabel: str,
    xlabel: str,
    column_labels: list[str],
):
    """
    Plot a crosstab using within-row percentages.

    Cell labels show:
        raw participant count
        within-row percentage
    """
    counts = counts.astype(int)
    row_totals = counts.sum(axis=1)

    percentages = (
        counts.div(
            row_totals.replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    image = ax.imshow(
        percentages.fillna(0).to_numpy(),
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    for row_index, workflow in enumerate(counts.index):
        row_total = int(row_totals.loc[workflow])

        for column_index, column in enumerate(counts.columns):
            count = int(counts.loc[workflow, column])

            if row_total == 0:
                continue

            percentage = percentages.loc[workflow, column]

            text_color = "white" if percentage >= 55 else "black"

            ax.text(
                column_index,
                row_index,
                f"{count}\n({percentage:.0f}%)",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(
        column_labels,
        rotation=28,
        ha="right",
    )

    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_yticklabels(
        [
            f"{workflow_display_name(workflow)} (n={int(row_totals.loc[workflow])})"
            for workflow in counts.index
        ]
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return image


def plot_stated_vs_observed_workflow_behaviour(
    ranking_rows: pd.DataFrame,
    main_df: pd.DataFrame,
) -> None:
    """
    Compare final Rank-1 preference with observed main-round choices.

    The first panel compares Rank 1 with the first voluntary choice.
    The second panel retains participants without a unique most-used workflow.
    """
    slug = "09_stated_vs_observed_workflow_behaviour"

    if ranking_rows.empty or main_df.empty:
        return

    stated_top = (
        ranking_rows[ranking_rows["rank"].eq(1)]
        .rename(columns={"workflow": "statedTop"})[["sessionId", "statedTop"]]
        .copy()
    )
    stated_top["sessionId"] = stated_top["sessionId"].astype(str)

    active_stated_workflows = [
        workflow
        for workflow in WORKFLOW_ORDER
        if stated_top["statedTop"].eq(workflow).any()
    ]

    # ------------------------------------------------------------
    # First voluntary choice
    # ------------------------------------------------------------
    first_choice = (
        main_df[main_df["roundIndex"].eq(MAIN_ROUND_INDICES[0])][
            ["sessionId", "workflow"]
        ]
        .drop_duplicates("sessionId", keep="last")
        .rename(columns={"workflow": "firstChoice"})
        .copy()
    )
    first_choice["sessionId"] = first_choice["sessionId"].astype(str)

    first_joined = stated_top.merge(
        first_choice,
        on="sessionId",
        how="inner",
        validate="one_to_one",
    )

    if first_joined.empty:
        return

    first_matrix = pd.crosstab(
        first_joined["statedTop"],
        first_joined["firstChoice"],
    ).reindex(
        index=active_stated_workflows,
        columns=WORKFLOW_ORDER,
        fill_value=0,
    )

    first_agreement = int(
        first_joined["statedTop"].eq(first_joined["firstChoice"]).sum()
    )
    first_n = len(first_joined)

    save_table(
        first_matrix,
        f"{slug}_first_choice_counts",
    )

    first_percentages = (
        first_matrix.div(
            first_matrix.sum(axis=1).replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    save_table(
        first_percentages,
        f"{slug}_first_choice_row_percentages",
    )

    # ------------------------------------------------------------
    # Most-used workflow, retaining ties
    # ------------------------------------------------------------
    sequences = _complete_main_sequences(main_df)

    session_by_participant = (
        main_df[["participantId", "sessionId"]]
        .dropna()
        .drop_duplicates("participantId", keep="last")
        .assign(sessionId=lambda df: df["sessionId"].astype(str))
        .set_index("participantId")["sessionId"]
    )

    tie_key = "__no_unique_modal__"
    modal_rows = []

    for participant_id, row in sequences.iterrows():
        counts = row.value_counts()
        top_workflows = counts[counts.eq(counts.max())].index.tolist()

        unique_modal = len(top_workflows) == 1

        modal_rows.append(
            {
                "participantId": participant_id,
                "modalChoice": (top_workflows[0] if unique_modal else tie_key),
                "hasUniqueModal": unique_modal,
            }
        )

    modal_choice = pd.DataFrame(modal_rows)

    if modal_choice.empty:
        return

    modal_choice["sessionId"] = modal_choice["participantId"].map(
        session_by_participant
    )

    modal_choice = modal_choice.dropna(subset=["sessionId"])

    modal_joined = stated_top.merge(
        modal_choice[
            [
                "sessionId",
                "modalChoice",
                "hasUniqueModal",
            ]
        ],
        on="sessionId",
        how="inner",
        validate="one_to_one",
    )

    modal_columns = [
        *WORKFLOW_ORDER,
        tie_key,
    ]

    modal_matrix = pd.crosstab(
        modal_joined["statedTop"],
        modal_joined["modalChoice"],
    ).reindex(
        index=active_stated_workflows,
        columns=modal_columns,
        fill_value=0,
    )

    modal_n = len(modal_joined)
    modal_ties = int((~modal_joined["hasUniqueModal"]).sum())

    unique_modal_joined = modal_joined[modal_joined["hasUniqueModal"]]

    unique_modal_n = len(unique_modal_joined)
    unique_modal_agreement = int(
        unique_modal_joined["statedTop"].eq(unique_modal_joined["modalChoice"]).sum()
    )

    overall_modal_matches = int(
        modal_joined["statedTop"].eq(modal_joined["modalChoice"]).sum()
    )

    save_table(
        modal_matrix,
        f"{slug}_modal_choice_counts",
    )

    modal_percentages = (
        modal_matrix.div(
            modal_matrix.sum(axis=1).replace(0, np.nan),
            axis=0,
        )
        * 100
    )

    save_table(
        modal_percentages,
        f"{slug}_modal_choice_row_percentages",
    )

    agreement_summary = pd.DataFrame(
        [
            {
                "comparison": ("Final Rank 1 versus first voluntary choice"),
                "participants": first_n,
                "agreementCount": first_agreement,
                "agreementPercentage": (first_agreement / first_n * 100),
                "ties": 0,
            },
            {
                "comparison": ("Final Rank 1 versus unique most-used workflow"),
                "participants": unique_modal_n,
                "agreementCount": unique_modal_agreement,
                "agreementPercentage": (
                    unique_modal_agreement / unique_modal_n * 100
                    if unique_modal_n
                    else np.nan
                ),
                "ties": modal_ties,
            },
            {
                "comparison": (
                    "Final Rank 1 versus most-used workflow, all participants"
                ),
                "participants": modal_n,
                "agreementCount": overall_modal_matches,
                "agreementPercentage": (
                    overall_modal_matches / modal_n * 100 if modal_n else np.nan
                ),
                "ties": modal_ties,
            },
        ]
    )

    save_table(
        agreement_summary,
        f"{slug}_agreement_summary",
        index=False,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (first_ax, modal_ax) = plt.subplots(
        1,
        2,
        figsize=(14.2, 5.6),
        layout="constrained",
    )

    first_image = _plot_row_percentage_crosstab(
        first_ax,
        first_matrix,
        (
            "First Voluntary Workflow Choice\n"
            f"Exact match: {first_agreement}/{first_n} "
            f"({first_agreement / first_n * 100:.0f}%)"
        ),
        "Final Rank-1 preference",
        "First voluntary choice",
        [workflow_display_name(workflow) for workflow in WORKFLOW_ORDER],
    )

    unique_agreement_text = (
        (
            f"{unique_modal_agreement}/{unique_modal_n} "
            f"({unique_modal_agreement / unique_modal_n * 100:.0f}%)"
        )
        if unique_modal_n
        else "not available"
    )

    _plot_row_percentage_crosstab(
        modal_ax,
        modal_matrix,
        (
            "Most-Used Workflow Across Main Rounds\n"
            f"Unique-mode agreement: {unique_agreement_text}; "
            f"{modal_ties} ties"
        ),
        "Final Rank-1 preference",
        "Observed most-used workflow",
        [
            *[workflow_display_name(workflow) for workflow in WORKFLOW_ORDER],
            "No unique\nmost-used workflow",
        ],
    )

    fig.colorbar(
        first_image,
        ax=[first_ax, modal_ax],
        label="Within stated-preference group (%)",
    )

    fig.suptitle(
        "Final Workflow Preference and Observed Main-Round Behaviour",
        fontsize=13,
    )

    save_figure(
        fig,
        slug,
        "Final Workflow Preference and Observed Main-Round Behaviour",
        (
            "Rows show final Rank-1 preference; cells show counts and row percentages for first and "
            "most-used main-round workflows."
        ),
    )


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------


def plot_workflow(df, feedback_df):
    """Generate the workflow-selection figures in analysis order."""
    main_df = phase_data(df, "main")
    if main_df.empty:
        return

    ranking_rows, audit_df = build_valid_ranking_rows(feedback_df)

    plot_total_workflow_usage_counts(main_df)
    plot_first_voluntary_workflow_choice(main_df)
    plot_workflow_distribution(main_df)
    plot_participant_workflow_trajectories(main_df)
    plot_practice_to_first_choice_transition(df)
    plot_main_workflow_transitions(main_df)
    plot_workflow_retention(main_df)
    plot_workflow_switching_behaviour(main_df)
    plot_final_workflow_preference(ranking_rows, audit_df)
    plot_stated_vs_observed_workflow_behaviour(ranking_rows, main_df)
