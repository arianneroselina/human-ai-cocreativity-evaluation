"""Print a compact completeness check for poem and evaluator data."""

from scripts.config import EXPECTED_EVALUATORS
from scripts.evaluation.database import fetch_all


EVALUATION_CHECK_QUERY = """
    SELECT
        (SELECT COUNT(*) FROM "Poem")::int AS "totalPoems",
        (
            SELECT COUNT(*) FROM "Poem" WHERE "isEmpty" = false
        )::int AS "nonEmptyPoems",
        (
            SELECT COUNT(*) FROM "Poem" WHERE "isEmpty" = true
        )::int AS "emptyPoems",
        (
            SELECT COUNT(*)
            FROM "Poem"
            WHERE "sessionId" IS NULL
               OR "participantId" IS NULL
               OR "roundIndex" IS NULL
        )::int AS "missingMetadata",
        (SELECT COUNT(*) FROM "EvaluationSession")::int AS "evaluatorSessions",
        (SELECT COUNT(*) FROM "Rating")::int AS "totalRatings";
"""


def main() -> None:
    """Query and print the evaluation-data completeness summary."""
    rows = fetch_all(EVALUATION_CHECK_QUERY)
    if not rows:
        print("Evaluation data check returned no result.")
        return

    summary = rows[0]
    expected_total_ratings = summary["nonEmptyPoems"] * EXPECTED_EVALUATORS

    print("\nEvaluation data check")
    print("=" * 80)
    print(f"Total poems:              {summary['totalPoems']}")
    print(f"Non-empty poems:          {summary['nonEmptyPoems']}")
    print(f"Empty poems:              {summary['emptyPoems']}")
    print(f"Missing poem metadata:    {summary['missingMetadata']}")
    print(f"Evaluator sessions:       {summary['evaluatorSessions']}")
    print(f"Total ratings:            {summary['totalRatings']}")
    print(f"Expected total ratings:   {expected_total_ratings}")


if __name__ == "__main__":
    main()
