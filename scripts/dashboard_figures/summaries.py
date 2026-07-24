"""Reusable descriptive summaries for dashboard figures.

This module contains statistics that were previously reimplemented in several
plot modules. The functions intentionally remain descriptive and preserve the
normal-approximation confidence intervals used by the original figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from scripts.config import CI_Z_VALUE


def grouped_metric_summary(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    *,
    ci_z_value: float = CI_Z_VALUE,
) -> pd.DataFrame:
    """Summarise one or more numeric metrics within each dataframe group.

    The returned column names match the existing round-series figures so this
    helper can replace their duplicated implementations without changing the
    exported table schema.
    """
    rows: list[dict[str, object]] = []

    for group_values, group_df in dataframe.groupby(
        list(group_columns),
        dropna=False,
        observed=True,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        group_values_dict = dict(zip(group_columns, group_values))

        for metric in metric_columns:
            if metric not in group_df.columns:
                continue

            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if values.empty:
                continue

            count = int(len(values))
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1)) if count > 1 else np.nan
            standard_error = (
                standard_deviation / np.sqrt(count) if count > 1 else np.nan
            )
            margin = (
                ci_z_value * standard_error if np.isfinite(standard_error) else np.nan
            )

            rows.append(
                {
                    **group_values_dict,
                    "metric": metric,
                    "mean": mean,
                    "median": float(values.median()),
                    "standardDeviation": standard_deviation,
                    "standardError": standard_error,
                    "count": count,
                    "lowerCI": mean - margin if np.isfinite(margin) else np.nan,
                    "upperCI": mean + margin if np.isfinite(margin) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def numeric_summary(
    values: pd.Series,
    *,
    ci_z_value: float = CI_Z_VALUE,
) -> dict[str, float | int]:
    """Return count, centre, spread, and a normal-approximation confidence interval."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    count = int(len(numeric))

    if count == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "se": np.nan,
            "ciLow": np.nan,
            "ciHigh": np.nan,
        }

    mean = float(numeric.mean())
    standard_deviation = float(numeric.std(ddof=1)) if count > 1 else np.nan
    standard_error = (
        standard_deviation / np.sqrt(count)
        if count > 1 and np.isfinite(standard_deviation)
        else np.nan
    )
    margin = ci_z_value * standard_error if np.isfinite(standard_error) else np.nan

    return {
        "count": count,
        "mean": mean,
        "median": float(numeric.median()),
        "std": standard_deviation,
        "se": standard_error,
        "ciLow": mean - margin if np.isfinite(margin) else np.nan,
        "ciHigh": mean + margin if np.isfinite(margin) else np.nan,
    }
