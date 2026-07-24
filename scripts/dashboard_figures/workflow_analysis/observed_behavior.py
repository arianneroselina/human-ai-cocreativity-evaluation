"""Stated workflow preference versus observed Main-round behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import MAIN_ROUND_INDICES, WORKFLOW_ORDER
from scripts.dashboard_figures.helpers import workflow_display_name
from scripts.utils import save_figure, save_table

from scripts.dashboard_figures.workflow_analysis.common import (
    _complete_main_sequences,
)


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
            "Rows show final Rank-1 preference; cells show counts and row "
            "percentages for first and "
            "most-used main-round workflows."
        ),
    )
