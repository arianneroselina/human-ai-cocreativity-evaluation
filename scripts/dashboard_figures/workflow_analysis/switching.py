"""Participant workflow-switching patterns across Main rounds."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import MAIN_ROUND_INDICES
from scripts.dashboard_figures.helpers import round_display_name
from scripts.dashboard_figures.style import (
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import save_figure, save_table

from scripts.dashboard_figures.workflow_analysis.common import (
    _complete_main_sequences,
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
