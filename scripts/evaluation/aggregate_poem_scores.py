import csv
import psycopg
from psycopg.rows import dict_row

from scripts.config import PRISMA_DATABASE_URL, EXPECTED_EVALUATORS, POEM_SCORES_PATH

query = """
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
        ORDER BY p."participantId", p."roundIndex"; \
        """

with psycopg.connect(PRISMA_DATABASE_URL, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

if not rows:
    print("No poem scores found. CSV was not created.")
    raise SystemExit(0)

with POEM_SCORES_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} poem scores to {POEM_SCORES_PATH}")
