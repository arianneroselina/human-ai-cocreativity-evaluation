"""Shared database and CSV helpers for evaluator-data scripts."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.config import require_database_url


def fetch_all(query: str) -> list[dict[str, Any]]:
    """Execute a read-only query and return all rows as dictionaries."""
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as error:
        raise RuntimeError(
            "Database scripts require psycopg. "
            "Install it with `pip install psycopg[binary]`."
        ) from error

    with psycopg.connect(require_database_url(), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return list(cursor.fetchall())


def write_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> bool:
    """Write dictionary rows to CSV and return whether a file was created."""
    if not rows:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return True
