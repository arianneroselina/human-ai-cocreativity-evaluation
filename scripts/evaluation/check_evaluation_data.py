import psycopg
from psycopg.rows import dict_row

from scripts.config import PRISMA_DATABASE_URL, EXPECTED_EVALUATORS


def print_table(title, rows):
    print(f"\n{title}")
    if not rows:
        print("No rows.")
        return

    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print("-" * 80)

    for row in rows:
        print(" | ".join(str(row[h]) for h in headers))


with psycopg.connect(PRISMA_DATABASE_URL, row_factory=dict_row) as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT COUNT(*) AS count FROM "Poem";')
        total_poems = cur.fetchone()["count"]

        cur.execute('SELECT COUNT(*) AS count FROM "Poem" WHERE "isEmpty" = false;')
        non_empty_poems = cur.fetchone()["count"]

        cur.execute('SELECT COUNT(*) AS count FROM "Poem" WHERE "isEmpty" = true;')
        empty_poems = cur.fetchone()["count"]

        cur.execute("""
            SELECT COUNT(*) AS count
            FROM "Poem"
            WHERE "sessionId" IS NULL
               OR "participantId" IS NULL
               OR "roundIndex" IS NULL;
        """)
        missing_metadata = cur.fetchone()["count"]

        cur.execute('SELECT COUNT(*) AS count FROM "EvaluationSession";')
        evaluator_sessions = cur.fetchone()["count"]

        cur.execute('SELECT COUNT(*) AS count FROM "Rating";')
        total_ratings = cur.fetchone()["count"]

        expected_total_ratings = non_empty_poems * EXPECTED_EVALUATORS

        print("\nEvaluation data check")
        print("=" * 80)
        print(f"Total poems:              {total_poems}")
        print(f"Non-empty poems:          {non_empty_poems}")
        print(f"Empty poems:              {empty_poems}")
        print(f"Missing poem metadata:    {missing_metadata}")
        print(f"Evaluator sessions:       {evaluator_sessions}")
        print(f"Total ratings:            {total_ratings}")
        print(f"Expected total ratings:   {expected_total_ratings}")

        cur.execute("""
            SELECT "workflow", COUNT(*) AS count
            FROM "Poem"
            GROUP BY "workflow"
            ORDER BY "workflow";
        """)
        print_table("Poems by workflow", cur.fetchall())

        cur.execute("""
            SELECT "roundIndex", COUNT(*) AS count
            FROM "Poem"
            GROUP BY "roundIndex"
            ORDER BY "roundIndex";
        """)
        print_table("Poems by round", cur.fetchall())

        cur.execute("""
            SELECT s."evaluatorId", COUNT(r."id") AS rating_count
            FROM "EvaluationSession" s
            LEFT JOIN "Rating" r ON r."sessionId" = s."id"
            GROUP BY s."evaluatorId"
            ORDER BY s."evaluatorId";
        """)
        print_table("Ratings by evaluator", cur.fetchall())

        cur.execute("""
            SELECT
                p."id" AS "poemId",
                p."participantId",
                p."roundIndex",
                p."workflow",
                COUNT(r."id") AS rating_count
            FROM "Poem" p
            LEFT JOIN "Rating" r ON r."poemId" = p."id"
            WHERE p."isEmpty" = false
            GROUP BY p."id", p."participantId", p."roundIndex", p."workflow"
            HAVING COUNT(r."id") <> %s
            ORDER BY rating_count ASC, p."participantId", p."roundIndex";
        """, (EXPECTED_EVALUATORS,))
        incomplete_poems = cur.fetchall()

        print_table(
            f"Poems not rated exactly {EXPECTED_EVALUATORS} times",
            incomplete_poems,
        )

        if missing_metadata == 0 and total_ratings == expected_total_ratings and not incomplete_poems:
            print("\nStatus: OK. Evaluation data looks complete.")
        else:
            print("\nStatus: Check needed. Some data is missing or incomplete.")
