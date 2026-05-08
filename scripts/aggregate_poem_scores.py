import csv
from pathlib import Path
import psycopg
from psycopg.rows import dict_row

from scripts.config import PRISMA_DATABASE_URL, EXPECTED_EVALUATORS

OUTPUT_PATH = Path("data/processed/poem_scores.csv")

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

processed_rows = []

for row in rows:
    means = [
        row["meanFluency"],
        row["meanThemeAlignment"],
        row["meanMeaningfulness"],
        row["meanPoeticness"],
        row["meanOverallQuality"],
    ]

    valid_means = [value for value in means if value is not None]

    row["qualityComposite"] = (
        sum(valid_means) / len(valid_means)
        if valid_means
        else None
    )

    row["isFullyRated"] = row["ratingCount"] == EXPECTED_EVALUATORS

    processed_rows.append(row)

if not processed_rows:
    print("No poem scores found. CSV was not created.")
    raise SystemExit(0)

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(processed_rows[0].keys()))
    writer.writeheader()
    writer.writerows(processed_rows)

print(f"Exported {len(processed_rows)} poem scores to {OUTPUT_PATH}")
