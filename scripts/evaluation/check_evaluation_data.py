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
