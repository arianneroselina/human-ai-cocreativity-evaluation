"""Quality and constraint-outcome summaries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.config import CI_Z_VALUE, QUALITY_PRIMARY_METRIC
from scripts.utils import parse_bool_or_none


def quality_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    metric: str = QUALITY_PRIMARY_METRIC,
) -> pd.DataFrame:
    """Calculate descriptive mean, spread, and normal-approximation 95% CI."""
    summary = (
        dataframe.groupby(group_columns, dropna=False)[metric]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ciLow"] = summary["mean"] - CI_Z_VALUE * summary["se"]
    summary["ciHigh"] = summary["mean"] + CI_Z_VALUE * summary["se"]
    summary.loc[summary["count"] < 2, ["std", "se", "ciLow", "ciHigh"]] = np.nan

    return summary


def wilson_interval(
    successes: int,
    total: int,
    z: float = CI_Z_VALUE,
) -> tuple[float, float]:
    """Return a 95% Wilson interval in percentage points."""
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0, (centre - margin) * 100), min(100, (centre + margin) * 100)


def pass_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarise complete constraint-pass rates with Wilson intervals."""
    data = dataframe.dropna(subset=["passedNumeric"]).copy()
    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby(group_columns)["passedNumeric"]
        .agg(totalRounds="count", passedRounds="sum")
        .reset_index()
    )
    summary["passedRounds"] = summary["passedRounds"].astype(int)
    summary["passRatePercent"] = summary["passedRounds"] / summary["totalRounds"] * 100

    intervals = summary.apply(
        lambda row: wilson_interval(
            int(row["passedRounds"]),
            int(row["totalRounds"]),
        ),
        axis=1,
        result_type="expand",
    )
    summary[["lowerCI", "upperCI"]] = intervals
    return summary


def parse_requirement_results(value) -> list[dict]:
    """Safely parse the JSON requirement-results field."""
    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []

    return parsed if isinstance(parsed, list) else []


def add_passed_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary complete-constraint outcome used by exposure figures."""
    prepared = df.copy()

    if "passedNumeric" in prepared.columns:
        prepared["passedNumeric"] = pd.to_numeric(
            prepared["passedNumeric"],
            errors="coerce",
        )
        return prepared

    if "passed" not in prepared.columns:
        prepared["passedNumeric"] = np.nan
        return prepared

    prepared["passedNumeric"] = prepared["passed"].apply(
        lambda value: (
            1.0
            if parse_bool_or_none(value) is True
            else 0.0
            if parse_bool_or_none(value) is False
            else np.nan
        )
    )

    return prepared
