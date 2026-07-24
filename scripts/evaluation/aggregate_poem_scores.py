"""Aggregate evaluator ratings into one score row per non-empty poem."""

from scripts.config import POEM_SCORES_PATH
from scripts.evaluation.database import fetch_all, write_csv_rows


POEM_SCORE_QUERY = """
    SELECT
        p."id" AS "poemId",
        p."sessionId",
        p."participantId",
        p."roundIndex",
        p."taskId",
        p."topic",
        p."workflow",
        p."timeMs",
        p."wordCount",
        p."charCount",
        p."passed",

        COUNT(r."id")::int AS "ratingCount",

        AVG(r."fluency")::float AS "meanFluency",
        AVG(r."themeAlignment")::float AS "meanThemeAlignment",
        AVG(r."meaningfulness")::float AS "meanMeaningfulness",
        AVG(r."poeticness")::float AS "meanPoeticness",
        AVG(r."overallQuality")::float AS "meanOverallQuality",

        STDDEV_SAMP(r."fluency")::float AS "sdFluency",
        STDDEV_SAMP(r."themeAlignment")::float AS "sdThemeAlignment",
        STDDEV_SAMP(r."meaningfulness")::float AS "sdMeaningfulness",
        STDDEV_SAMP(r."poeticness")::float AS "sdPoeticness",
        STDDEV_SAMP(r."overallQuality")::float AS "sdOverallQuality"
    FROM "Poem" p
    LEFT JOIN "Rating" r ON r."poemId" = p."id"
    WHERE p."isEmpty" = false
    GROUP BY
        p."id",
        p."sessionId",
        p."participantId",
        p."roundIndex",
        p."taskId",
        p."topic",
        p."workflow",
        p."timeMs",
        p."wordCount",
        p."charCount",
        p."passed"
    ORDER BY p."participantId", p."roundIndex";
"""


def main() -> None:
    """Query, aggregate, and export poem scores."""
    rows = fetch_all(POEM_SCORE_QUERY)
    if not write_csv_rows(rows, POEM_SCORES_PATH):
        print("No poem scores found. CSV was not created.")
        return

    print(f"Exported {len(rows)} poem scores to {POEM_SCORES_PATH}")


if __name__ == "__main__":
    main()
