import csv
from pathlib import Path
import psycopg
from psycopg.rows import dict_row

from scripts.config import PRISMA_DATABASE_URL

OUTPUT_PATH = Path("data/processed/ratings_export.csv")

query = """
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

with psycopg.connect(PRISMA_DATABASE_URL, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

if not rows:
    print("No ratings found. CSV was not created.")
    raise SystemExit(0)

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} ratings to {OUTPUT_PATH}")
