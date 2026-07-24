"""Practice-to-Main and Main-round workflow transition figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    MAIN_ROUND_INDICES,
    PRACTICE_ROUND_INDICES,
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
    apply_standard_axes_style,
)
from scripts.utils import save_figure, save_table

from scripts.dashboard_figures.workflow_analysis.common import (
    _plot_transition_heatmap,
    _transition_matrix,
    _transition_rows_for_pair,
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
        "Each cell shows the participant count and the percentage within the "
        "workflow selected in the preceding Main round.",
    )


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
    ax.tick_params(axis="x")

    save_figure(
        fig,
        slug,
        "Workflow Retention in the Next Main Round",
        "Probability of choosing the same workflow in the immediately following "
        "main round.",
    )
