"""Post-error workflow choices by first Main-round exposure."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import (
    INJECTED_ERROR_ROUND_INDEX,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import (
    exposure_display_name,
    round_display_name,
    workflow_display_name,
)
from scripts.dashboard_figures.style import (
    WORKFLOW_COLORS,
    BAR_EDGE_COLOR,
    apply_standard_axes_style,
)
from scripts.utils import (
    save_figure,
    save_table,
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
        ax.set_xlabel("Post-error in main rounds")
        ax.set_title(exposure_display_name(group))
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
        "Post-Error Workflow Choices by Main Round 1 Exposure", fontsize=13, y=0.99
    )

    fig.tight_layout(rect=(0, 0.045, 0.84, 0.96))

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Main Round 1 Exposure",
        "Distribution of voluntary workflow choices in Main Rounds 2-3 by "
        "error exposure.",
    )
