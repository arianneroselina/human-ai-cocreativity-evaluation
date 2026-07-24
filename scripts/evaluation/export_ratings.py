"""Export raw evaluator ratings with the associated poem metadata."""

from scripts.config import RATINGS_EXPORT_PATH
from scripts.evaluation.database import fetch_all, write_csv_rows


RATINGS_EXPORT_QUERY = """
    SELECT
        r."id" AS "ratingId",
        r."poemId",
        s."evaluatorId",

        p."sessionId" AS "participantSessionId",
        p."participantId",
        p."roundIndex",
        p."taskId",
        p."topic",
        p."workflow",

        p."timeMs",
        p."wordCount",
        p."charCount",
        p."passed",

        r."fluency",
        r."themeAlignment",
        r."meaningfulness",
        r."poeticness",
        r."overallQuality",
        r."comment",
        r."timeSpentMs",

        r."createdAt",
        r."updatedAt"
    FROM "Rating" r
    JOIN "Poem" p ON p."id" = r."poemId"
    JOIN "EvaluationSession" s ON s."id" = r."sessionId"
    ORDER BY p."participantId", p."roundIndex", s."evaluatorId";
"""


def main() -> None:
    """Query and export all raw ratings."""
    rows = fetch_all(RATINGS_EXPORT_QUERY)
    if not write_csv_rows(rows, RATINGS_EXPORT_PATH):
        print("No ratings found. CSV was not created.")
        return

    print(f"Exported {len(rows)} ratings to {RATINGS_EXPORT_PATH}")


if __name__ == "__main__":
    main()
