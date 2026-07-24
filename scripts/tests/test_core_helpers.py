"""Regression tests for the refactored data and plotting helpers."""

from __future__ import annotations

import json
import os
import unittest

import numpy as np
import pandas as pd

from scripts.create_master_dataset import (
    add_error_exposure_columns,
    safe_rate,
)
from scripts.dashboard_figures.helpers import (
    build_valid_ranking_rows,
    evaluator_color,
    main_round_tick_labels,
)
from scripts.dashboard_figures.summaries import grouped_metric_summary


class ConfigurationImportTests(unittest.TestCase):
    def test_config_import_does_not_require_database_url(self) -> None:
        previous = os.environ.pop("PRISMA_DATABASE_URL", None)
        try:
            import scripts.config as config

            self.assertTrue(hasattr(config, "require_database_url"))
        finally:
            if previous is not None:
                os.environ["PRISMA_DATABASE_URL"] = previous


class LabelTests(unittest.TestCase):
    def test_main_round_labels_mark_only_injected_error_round(self) -> None:
        self.assertEqual(
            main_round_tick_labels(
                [5, 6, 7],
                mark_injected_error=True,
            ),
            ["Main 1\nInjected AI error", "Main 2", "Main 3"],
        )


class EvaluatorStyleTests(unittest.TestCase):
    def test_unexpected_evaluator_ids_receive_stable_colors(self) -> None:
        self.assertEqual(
            evaluator_color("unexpected-evaluator"),
            evaluator_color("unexpected-evaluator"),
        )
        self.assertTrue(evaluator_color("unexpected-evaluator").startswith("#"))


class SummaryTests(unittest.TestCase):
    def test_grouped_metric_summary_preserves_singleton_ci_as_missing(self) -> None:
        dataframe = pd.DataFrame(
            {
                "workflow": ["human", "human", "ai"],
                "roundIndex": [1, 1, 1],
                "score": [2.0, 4.0, 5.0],
            }
        )

        summary = grouped_metric_summary(
            dataframe,
            group_columns=["roundIndex", "workflow"],
            metric_columns=["score"],
        )

        human = summary.loc[summary["workflow"].eq("human")].iloc[0]
        ai = summary.loc[summary["workflow"].eq("ai")].iloc[0]
        self.assertEqual(human["mean"], 3.0)
        self.assertEqual(human["count"], 2)
        self.assertTrue(np.isnan(ai["lowerCI"]))
        self.assertTrue(np.isnan(ai["upperCI"]))


class MasterDatasetHelperTests(unittest.TestCase):
    def test_missing_main_round_one_is_not_classified_as_not_exposed(self) -> None:
        dataframe = pd.DataFrame(
            {
                "participantId": ["p1", "p1", "p2"],
                "roundIndex": [5, 6, 6],
                "workflow": ["ai", "human", "human"],
            }
        )

        prepared = add_error_exposure_columns(dataframe)
        p1 = prepared.loc[prepared["participantId"].eq("p1"), "errorExposed"]
        p2 = prepared.loc[prepared["participantId"].eq("p2"), "errorExposed"]
        self.assertTrue(p1.fillna(False).all())
        self.assertTrue(p2.isna().all())

    def test_safe_rate_treats_zero_time_as_missing(self) -> None:
        result = safe_rate(
            pd.Series([10.0, 10.0]),
            pd.Series([2.0, 0.0]),
        )
        self.assertEqual(result.iloc[0], 5.0)
        self.assertTrue(np.isnan(result.iloc[1]))


class RankingTests(unittest.TestCase):
    def test_complete_rankings_are_expanded_and_invalid_rankings_are_audited(
        self,
    ) -> None:
        feedback = pd.DataFrame(
            {
                "sessionId": ["s1", "s2"],
                "workflowRanking": [
                    json.dumps(["human", "ai", "human_ai", "ai_human"]),
                    json.dumps(["human", "ai"]),
                ],
            }
        )

        ranking_rows, audit = build_valid_ranking_rows(feedback)
        self.assertEqual(len(ranking_rows), 4)
        self.assertEqual(int(audit["validRanking"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
