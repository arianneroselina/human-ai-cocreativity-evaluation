"""Shared workflow-sequence and transition helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import (
    MAIN_ROUND_INDICES,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import workflow_display_name


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
