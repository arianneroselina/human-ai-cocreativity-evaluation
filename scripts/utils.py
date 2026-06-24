"""Shared utility functions for dashboard data preparation and figure export."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.config import (
    FIGURE_DIR,
    TABLE_DIR,
)


MANIFEST = []


def reset_manifest() -> None:
    """Clear previously registered figures in the current Python process."""
    MANIFEST.clear()


def require_columns(
    df: pd.DataFrame,
    required_columns,
    context: str = "dataframe",
) -> bool:
    """Return whether a dataframe contains all required columns."""
    missing = set(required_columns) - set(df.columns)

    if missing:
        print(f"Skipping {context}; missing columns: {sorted(missing)}")
        return False

    return True


def ensure_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Convert available dataframe columns to numeric values in place."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def parse_bool(value) -> bool:
    """Parse common boolean-like values; missing or unknown values become False."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    return normalized in {"true", "t", "1", "yes", "y"}


def parse_bool_or_none(value):
    """Parse common boolean-like values; missing or unknown values become None."""
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "t", "1", "yes", "y"}:
        return True

    if normalized in {"false", "f", "0", "no", "n"}:
        return False

    return None


def drop_duplicate_participant_rounds(
    df: pd.DataFrame,
    participant_column: str = "participantId",
    round_column: str = "roundIndex",
) -> pd.DataFrame:
    """Keep the first row for each participant-round combination."""
    required_columns = {participant_column, round_column}

    if not required_columns.issubset(df.columns):
        return df

    return df.drop_duplicates(
        subset=[participant_column, round_column],
        keep="first",
    )


def format_count_percentage(
    count: int | float,
    total: int | float,
    decimals: int = 0,
) -> str:
    """Format a count with its percentage of a total."""
    if not total:
        return f"{int(count)} (n/a)"

    percentage = count / total * 100
    return f"{int(count)} ({percentage:.{decimals}f}%)"


def save_table(
    table: pd.DataFrame | pd.Series,
    slug: str,
    index: bool = True,
) -> Path:
    """Export a dataframe or series as a CSV table."""
    table_path = TABLE_DIR / f"{slug}.csv"
    table.to_csv(table_path, index=index)

    return table_path


def save_figure(
    fig,
    slug: str,
    title: str,
    description: str,
) -> None:
    """Save one figure in dashboard formats and register it in the manifest."""
    png_path = FIGURE_DIR / f"{slug}.png"
    pdf_path = FIGURE_DIR / f"{slug}.pdf"
    svg_path = FIGURE_DIR / f"{slug}.svg"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    MANIFEST.append(
        {
            "slug": slug,
            "title": title,
            "description": description,
            "pngUrl": f"/research-dashboard/figures/{slug}.png",
            "pdfUrl": f"/research-dashboard/figures/{slug}.pdf",
            "svgUrl": f"/research-dashboard/figures/{slug}.svg",
        }
    )


def save_manifest() -> Path:
    """Write the registered figure metadata for the Next.js dashboard."""
    manifest_path = FIGURE_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(MANIFEST, file, indent=2, ensure_ascii=False)

    return manifest_path
