"""Shared preparation and summaries for subjective-experience figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.config import (
    AI_EXPERIENCE_METRICS,
    QUALITY_PRIMARY_METRIC,
    SATISFACTION_COLUMN,
    TLX_METRICS,
)
from scripts.dashboard_figures.helpers import phase_data


def _prepare_experience_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert experience-analysis variables to numeric values."""
    prepared = df.copy()

    numeric_columns = [
        SATISFACTION_COLUMN,
        QUALITY_PRIMARY_METRIC,
        *AI_EXPERIENCE_METRICS,
        *TLX_METRICS,
    ]

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    return prepared


def _available_rounds(dataframe: pd.DataFrame) -> list[int]:
    """Return observed study rounds in ascending order."""
    return sorted(dataframe["roundIndex"].dropna().astype(int).unique().tolist())


def _spearman_summary(
    dataframe: pd.DataFrame,
    phase: str,
) -> dict[str, float | int | str]:
    """Return a descriptive pooled Spearman association for one study phase."""
    phase_data = dataframe[dataframe["phase"].eq(phase)].dropna(
        subset=[SATISFACTION_COLUMN, QUALITY_PRIMARY_METRIC]
    )

    rho = phase_data[SATISFACTION_COLUMN].corr(
        phase_data[QUALITY_PRIMARY_METRIC],
        method="spearman",
    )

    return {
        "phase": phase,
        "observations": int(len(phase_data)),
        "spearmanRho": float(rho) if pd.notna(rho) else np.nan,
        "meanSatisfaction": (
            float(phase_data[SATISFACTION_COLUMN].mean())
            if not phase_data.empty
            else np.nan
        ),
        "meanQuality": (
            float(phase_data[QUALITY_PRIMARY_METRIC].mean())
            if not phase_data.empty
            else np.nan
        ),
    }
