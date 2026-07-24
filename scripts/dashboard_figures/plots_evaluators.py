"""Generate overall-quality evaluator agreement figures.

Preparation, distributions, reliability, disagreement, and matrices are split
into focused modules under ``dashboard_figures.evaluator_analysis``.
"""

from __future__ import annotations

import pandas as pd

from scripts.dashboard_figures.evaluator_analysis import (
    plot_evaluator_disagreement_magnitude,
    plot_evaluator_rating_tendency,
    plot_overall_quality_icc_reliability,
    plot_overall_quality_ordinal_agreement,
    plot_overall_quality_rating_distribution,
    plot_pairwise_overall_quality_agreement,
    plot_pairwise_overall_quality_matrices,
)
from scripts.dashboard_figures.evaluator_analysis.common import (
    _complete_rating_panel,
    _load_ratings,
    _pairwise_summary,
    _prepare_ratings,
)


def plot_evaluators(ratings_df: pd.DataFrame | None = None) -> None:
    """Generate overall-quality evaluator-agreement dashboard outputs."""
    if ratings_df is None:
        ratings_df = _load_ratings()

    prepared = _prepare_ratings(ratings_df)
    if prepared.empty:
        return

    wide_df, evaluators, total_poems = _complete_rating_panel(prepared)
    if wide_df.empty or len(evaluators) < 2:
        print(
            "Skipping evaluator figures; fewer than two evaluators rated a "
            "complete set of overall-quality scores."
        )
        return

    if len(wide_df) != total_poems:
        print(
            "Evaluator figures use only poems with a complete rating panel: "
            f"{len(wide_df)}/{total_poems} poems."
        )

    plot_overall_quality_rating_distribution(wide_df, evaluators)
    plot_overall_quality_ordinal_agreement(wide_df, evaluators)
    plot_overall_quality_icc_reliability(wide_df, evaluators)

    pairwise_df = _pairwise_summary(wide_df, evaluators)
    plot_pairwise_overall_quality_agreement(
        wide_df,
        evaluators,
        pairwise_df=pairwise_df,
    )
    plot_evaluator_disagreement_magnitude(wide_df, evaluators)
    plot_evaluator_rating_tendency(wide_df, evaluators)
    plot_pairwise_overall_quality_matrices(
        wide_df,
        evaluators,
        pairwise_df=pairwise_df,
    )
