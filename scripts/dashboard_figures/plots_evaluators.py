"""Evaluator agreement and calibration figures.

Figures
-------
61  ICC(A,1) and ICC(A,k) with 95% confidence intervals
62  Pairwise ICC(A,1) diagnostic heatmaps
63  Evaluator calibration relative to peer ratings
64  Supplementary ordinal pairwise agreement metrics

A quadratic-weighted Cohen's Kappa summary is also exported as CSV as a
supplementary robustness check, without creating an additional dashboard figure.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

from scripts.config import (
    EVALUATOR_LABELS,
    EVALUATOR_ORDER,
    RATINGS_EXPORT_PATH,
    RATING_DIMENSIONS,
    RATING_DIMENSION_LABELS,
)
from scripts.dashboard_figures.helpers import evaluator_display_name
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import require_columns, save_figure, save_table, save_analysis_table


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _ordered_evaluators(ratings_df: pd.DataFrame) -> list[str]:
    """Return known evaluators first, followed by any additional evaluators."""
    available = set(ratings_df["evaluatorId"].dropna().astype(str))
    ordered = [evaluator for evaluator in EVALUATOR_ORDER if evaluator in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _available_dimensions(ratings_df: pd.DataFrame) -> list[str]:
    """Return configured dimensions present in the supplied ratings export."""
    return [
        dimension
        for dimension in RATING_DIMENSIONS
        if dimension in ratings_df.columns
           and pd.to_numeric(ratings_df[dimension], errors="coerce").notna().any()
    ]


def _load_ratings() -> pd.DataFrame:
    """Load the raw evaluator ratings export when no dataframe is supplied."""
    if not RATINGS_EXPORT_PATH.exists():
        print(f"Skipping evaluator figures; ratings export not found: {RATINGS_EXPORT_PATH}")
        return pd.DataFrame()

    return pd.read_csv(RATINGS_EXPORT_PATH)


def _prepare_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize evaluator ratings used throughout this module."""
    required = {"poemId", "evaluatorId"}
    if ratings_df.empty or not require_columns(ratings_df, required, "evaluator ratings"):
        return pd.DataFrame()

    prepared = ratings_df.copy()
    prepared["evaluatorId"] = prepared["evaluatorId"].astype(str)

    for dimension in RATING_DIMENSIONS:
        if dimension in prepared.columns:
            prepared[dimension] = pd.to_numeric(prepared[dimension], errors="coerce")

    # One rating per poem and evaluator is expected. Keep the final row if an
    # export contains a duplicate, while retaining all original columns.
    prepared = prepared.drop_duplicates(
        subset=["poemId", "evaluatorId"],
        keep="last",
    )

    return prepared


def _balanced_dimension_long(
        ratings_df: pd.DataFrame,
        dimension: str,
        evaluators: list[str] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Return a complete poem-by-evaluator panel for one dimension.

    ICC requires a balanced panel. Rows with a missing rating from any included
    evaluator are removed explicitly, rather than relying on implicit handling
    inside the ICC implementation.
    """
    if dimension not in ratings_df.columns:
        return pd.DataFrame(), 0, 0

    long_df = ratings_df[["poemId", "evaluatorId", dimension]].dropna().copy()
    long_df = long_df.rename(columns={dimension: "rating"})

    if evaluators is not None:
        long_df = long_df[long_df["evaluatorId"].isin(evaluators)]

    if long_df.empty:
        return pd.DataFrame(), 0, 0

    wide_df = (
        long_df.pivot(index="poemId", columns="evaluatorId", values="rating")
        .dropna(how="any")
    )

    if wide_df.shape[0] < 2 or wide_df.shape[1] < 2:
        return pd.DataFrame(), 0, 0

    balanced_long = (
        wide_df.rename_axis(index="poemId", columns="evaluatorId")
        .stack()
        .rename("rating")
        .reset_index()
    )

    return balanced_long, int(wide_df.shape[0]), int(wide_df.shape[1])


# ---------------------------------------------------------------------------
# Reliability calculations
# ---------------------------------------------------------------------------


def _parse_ci(value) -> tuple[float, float] | None:
    """Parse Pingouin confidence intervals across supported package versions."""
    if value is None:
        return None

    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        return None

    try:
        lower, upper = value[0], value[1]
        return float(lower), float(upper)
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _select_icc_row(icc_table: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    """Select an ICC row regardless of Pingouin's version-specific labels."""
    for alias in aliases:
        match = icc_table[icc_table["Type"].eq(alias)]
        if not match.empty:
            return match.iloc[0]
    return None


def _compute_icc(
        balanced_long: pd.DataFrame,
        n_poems: int,
        n_evaluators: int,
) -> dict[str, dict[str, float]] | None:
    """Compute absolute-agreement ICC for a single and averaged rater score.

    ICC(A,1) / ICC2: reliability of one individual evaluator.
    ICC(A,k) / ICC2k: reliability of the average across the full evaluator panel.
    """
    if balanced_long.empty or n_poems < 2 or n_evaluators < 2:
        return None

    try:
        icc_table = pg.intraclass_corr(
            data=balanced_long,
            targets="poemId",
            raters="evaluatorId",
            ratings="rating",
        )
    except Exception as error:
        print(f"Unable to calculate ICC: {error}")
        return None

    result = {}
    requested_types = {
        "ICC(A,1)": ["ICC(A,1)", "ICC2"],
        "ICC(A,k)": ["ICC(A,k)", "ICC2k"],
    }

    for display_name, aliases in requested_types.items():
        row = _select_icc_row(icc_table, aliases)
        if row is None:
            continue

        ci = _parse_ci(row.get("CI95%", row.get("CI95", None)))
        if ci is None:
            continue

        result[display_name] = {
            "icc": float(row["ICC"]),
            "lower_ci": ci[0],
            "upper_ci": ci[1],
            "f_value": float(row.get("F", np.nan)),
            "p_value": float(row.get("pval", np.nan)),
        }

    return result or None


def _icc_summary(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ICC(A,1) and ICC(A,k) for every available rating dimension."""
    evaluators = _ordered_evaluators(ratings_df)
    rows = []

    for dimension in _available_dimensions(ratings_df):
        balanced_long, n_poems, n_evaluators = _balanced_dimension_long(
            ratings_df,
            dimension,
            evaluators=evaluators,
        )
        results = _compute_icc(balanced_long, n_poems, n_evaluators)
        if results is None:
            continue

        for icc_type, values in results.items():
            rows.append(
                {
                    "dimension": dimension,
                    "dimensionLabel": RATING_DIMENSION_LABELS.get(dimension, dimension),
                    "iccType": icc_type,
                    "icc": values["icc"],
                    "lowerCI": values["lower_ci"],
                    "upperCI": values["upper_ci"],
                    "fValue": values["f_value"],
                    "pValue": values["p_value"],
                    "nPoems": n_poems,
                    "nEvaluators": n_evaluators,
                }
            )

    return pd.DataFrame(rows)


def _pairwise_icc_summary(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise ICC(A,1) for all dimensions and evaluator pairs."""
    evaluators = _ordered_evaluators(ratings_df)
    rows = []

    for dimension in _available_dimensions(ratings_df):
        for evaluator_a, evaluator_b in combinations(evaluators, 2):
            pair_long, n_poems, n_evaluators = _balanced_dimension_long(
                ratings_df,
                dimension,
                evaluators=[evaluator_a, evaluator_b],
            )
            results = _compute_icc(pair_long, n_poems, n_evaluators)
            if results is None or "ICC(A,1)" not in results:
                continue

            values = results["ICC(A,1)"]
            rows.append(
                {
                    "dimension": dimension,
                    "dimensionLabel": RATING_DIMENSION_LABELS.get(dimension, dimension),
                    "evaluatorA": evaluator_a,
                    "evaluatorALabel": evaluator_display_name(evaluator_a),
                    "evaluatorB": evaluator_b,
                    "evaluatorBLabel": evaluator_display_name(evaluator_b),
                    "pairLabel": f"{evaluator_display_name(evaluator_a)} vs {evaluator_display_name(evaluator_b)}",
                    "iccA1": values["icc"],
                    "lowerCI": values["lower_ci"],
                    "upperCI": values["upper_ci"],
                    "nSharedPoems": n_poems,
                }
            )

    return pd.DataFrame(rows)


def _weighted_kappa_summary(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ordinal pairwise agreement for supplementary reporting.

    Quadratic weights respect that 4 vs. 5 is a smaller disagreement than
    1 vs. 5 on the ordinal 1–5 rating scale.
    """
    evaluators = _ordered_evaluators(ratings_df)
    rows = []

    for dimension in _available_dimensions(ratings_df):
        for evaluator_a, evaluator_b in combinations(evaluators, 2):
            pair_long, n_poems, _ = _balanced_dimension_long(
                ratings_df,
                dimension,
                evaluators=[evaluator_a, evaluator_b],
            )
            if pair_long.empty:
                continue

            pivot = pair_long.pivot(
                index="poemId",
                columns="evaluatorId",
                values="rating",
            )
            ratings_a = pivot[evaluator_a].astype(int)
            ratings_b = pivot[evaluator_b].astype(int)

            rows.append(
                {
                    "dimension": dimension,
                    "dimensionLabel": RATING_DIMENSION_LABELS.get(dimension, dimension),
                    "pairLabel": f"{evaluator_display_name(evaluator_a)} vs {evaluator_display_name(evaluator_b)}",
                    "quadraticWeightedKappa": cohen_kappa_score(
                        ratings_a,
                        ratings_b,
                        weights="quadratic",
                    ),
                    "exactAgreementPercentage": (ratings_a == ratings_b).mean() * 100,
                    "withinOnePointPercentage": (
                            (ratings_a.sub(ratings_b).abs() <= 1).mean() * 100
                    ),
                    "nSharedPoems": n_poems,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 61 – ICC(A,1) and ICC(A,k)
# ---------------------------------------------------------------------------


def plot_icc_confidence_intervals(ratings_df: pd.DataFrame) -> None:
    """Plot single-rater and averaged-panel ICC estimates with 95% CIs."""
    slug = "61_inter_rater_agreement_icc"
    summary_df = _icc_summary(ratings_df)

    if summary_df.empty:
        return

    save_table(summary_df, slug, index=False)

    dimension_order = [
        RATING_DIMENSION_LABELS.get(dimension, dimension)
        for dimension in _available_dimensions(ratings_df)
        if dimension in set(summary_df["dimension"])
    ]
    summary_df = summary_df.copy()
    summary_df["dimensionLabel"] = pd.Categorical(
        summary_df["dimensionLabel"],
        categories=dimension_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("dimensionLabel")

    fig, ax = plt.subplots(figsize=(9.2, max(4.6, 0.78 * len(dimension_order) + 2.2)))
    y_base = np.arange(len(dimension_order))
    offsets = {"ICC(A,1)": -0.12, "ICC(A,k)": 0.12}
    markers = {"ICC(A,1)": "o", "ICC(A,k)": "s"}

    for icc_type in ["ICC(A,1)", "ICC(A,k)"]:
        subset = (
            summary_df[summary_df["iccType"] == icc_type]
            .set_index("dimensionLabel")
            .reindex(dimension_order)
            .dropna(subset=["icc"])
        )
        if subset.empty:
            continue

        positions = np.array([dimension_order.index(label) for label in subset.index])
        lower_error = subset["icc"].values - subset["lowerCI"].values
        upper_error = subset["upperCI"].values - subset["icc"].values

        ax.errorbar(
            subset["icc"].values,
            positions + offsets[icc_type],
            xerr=np.vstack([lower_error, upper_error]),
            fmt=markers[icc_type],
            capsize=4,
            markersize=7,
            label=icc_type,
            )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(0.50, linestyle=":", linewidth=1, alpha=0.8)
    ax.axvline(0.75, linestyle=":", linewidth=1, alpha=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels(dimension_order)
    ax.set_xlabel("Intraclass correlation coefficient")
    ax.set_ylabel("Rating dimension")
    ax.set_title("Inter-Rater Reliability by Quality Dimension")
    ax.legend(title="Reliability target", loc="lower right")
    apply_standard_axes_style(ax, grid_axis="x")

    lower_bound = min(-0.1, float(summary_df["lowerCI"].min()) - 0.05)
    upper_bound = max(1.0, float(summary_df["upperCI"].max()) + 0.05)
    ax.set_xlim(lower_bound, upper_bound)

    save_figure(
        fig,
        slug,
        "Inter-Rater Reliability by Quality Dimension",
        "ICC(A,1) estimates the agreement of one individual evaluator; ICC(A,k) "
        "estimates the reliability of the averaged score across the full evaluator panel. "
        "Error bars show 95% confidence intervals.",
    )


# ---------------------------------------------------------------------------
# 2 – Pairwise ICC(A,1) diagnostic heatmaps
# ---------------------------------------------------------------------------


def plot_pairwise_icc_heatmaps(ratings_df: pd.DataFrame) -> None:
    """Show pairwise single-rater ICCs for diagnostic comparison."""
    slug = "62_pairwise_icc_heatmaps"
    pairwise_df = _pairwise_icc_summary(ratings_df)
    evaluators = _ordered_evaluators(ratings_df)

    if pairwise_df.empty or len(evaluators) < 2:
        return

    save_table(pairwise_df, slug, index=False)

    dimensions = [
        dimension
        for dimension in _available_dimensions(ratings_df)
        if dimension in set(pairwise_df["dimension"])
    ]
    n_columns = min(3, len(dimensions))
    n_rows = int(np.ceil(len(dimensions) / n_columns))

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4.5 * n_columns, 4.0 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    images = []

    for axis, dimension in zip(axes_flat, dimensions):
        matrix = pd.DataFrame(np.nan, index=evaluators, columns=evaluators)
        overlap = pd.DataFrame(np.nan, index=evaluators, columns=evaluators)
        dimension_df = pairwise_df[pairwise_df["dimension"] == dimension]

        for _, row in dimension_df.iterrows():
            evaluator_a = row["evaluatorA"]
            evaluator_b = row["evaluatorB"]
            matrix.loc[evaluator_a, evaluator_b] = row["iccA1"]
            matrix.loc[evaluator_b, evaluator_a] = row["iccA1"]
            overlap.loc[evaluator_a, evaluator_b] = row["nSharedPoems"]
            overlap.loc[evaluator_b, evaluator_a] = row["nSharedPoems"]

        masked_values = np.ma.masked_invalid(matrix.values)
        image = axis.imshow(masked_values, vmin=-0.25, vmax=1.0, cmap="RdYlGn")
        images.append(image)

        for row_index, evaluator_a in enumerate(evaluators):
            for col_index, evaluator_b in enumerate(evaluators):
                if row_index >= col_index:
                    axis.text(col_index, row_index, "–", ha="center", va="center", fontsize=11)
                    continue

                value = matrix.iloc[row_index, col_index]
                n_shared = overlap.iloc[row_index, col_index]
                label = "–" if pd.isna(value) else f"{value:.2f}"
                text_color = "white" if not pd.isna(value) and value >= 0.62 else "black"
                axis.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                )

        axis.set_xticks(range(len(evaluators)))
        axis.set_xticklabels([evaluator_display_name(evaluator) for evaluator in evaluators], rotation=25, ha="right")
        axis.set_yticks(range(len(evaluators)))
        axis.set_yticklabels([evaluator_display_name(evaluator) for evaluator in evaluators])
        axis.set_title(RATING_DIMENSION_LABELS.get(dimension, dimension))

    for axis in axes_flat[len(dimensions):]:
        axis.set_visible(False)

    fig.colorbar(images[0], ax=[axis for axis in axes_flat[:len(dimensions)]], label="Pairwise ICC(A,1)")
    fig.suptitle("Pairwise Single-Rater Agreement", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))

    save_figure(
        fig,
        slug,
        "Pairwise Single-Rater Agreement",
        "Upper-triangle pairwise ICC(A,1) values across evaluator pairs. "
        "All pairs rated the same 166 poems.",
    )


# ---------------------------------------------------------------------------
# 63 – Evaluator calibration relative to peers
# ---------------------------------------------------------------------------


def _calibration_summary(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each evaluator's deviation from the mean of the other raters."""
    evaluators = _ordered_evaluators(ratings_df)
    rows = []

    for dimension in _available_dimensions(ratings_df):
        balanced_long, _, _ = _balanced_dimension_long(
            ratings_df,
            dimension,
            evaluators=evaluators,
        )
        if balanced_long.empty:
            continue

        pivot = balanced_long.pivot(
            index="poemId",
            columns="evaluatorId",
            values="rating",
        ).reindex(columns=evaluators)

        for evaluator in evaluators:
            peers = [peer for peer in evaluators if peer != evaluator]
            deviations = pivot[evaluator] - pivot[peers].mean(axis=1)
            deviations = deviations.dropna()

            if deviations.empty:
                continue

            n = len(deviations)
            mean_deviation = float(deviations.mean())
            standard_error = float(deviations.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            margin = 1.96 * standard_error if np.isfinite(standard_error) else np.nan

            rows.append(
                {
                    "dimension": dimension,
                    "dimensionLabel": RATING_DIMENSION_LABELS.get(dimension, dimension),
                    "evaluatorId": evaluator,
                    "evaluatorLabel": evaluator_display_name(evaluator),
                    "meanDeviationFromPeers": mean_deviation,
                    "lowerCI": mean_deviation - margin if np.isfinite(margin) else np.nan,
                    "upperCI": mean_deviation + margin if np.isfinite(margin) else np.nan,
                    "nPoems": n,
                }
            )

    return pd.DataFrame(rows)


def plot_evaluator_calibration(ratings_df: pd.DataFrame) -> None:
    """Show whether each evaluator rates above or below their peer average."""
    slug = "63_evaluator_calibration_relative_to_peers"
    calibration_df = _calibration_summary(ratings_df)

    if calibration_df.empty:
        return

    save_table(calibration_df, slug, index=False)

    dimensions = [
        RATING_DIMENSION_LABELS.get(dimension, dimension)
        for dimension in _available_dimensions(ratings_df)
        if dimension in set(calibration_df["dimension"])
    ]
    evaluators = _ordered_evaluators(ratings_df)
    y_base = np.arange(len(dimensions))
    offsets = np.linspace(-0.18, 0.18, len(evaluators))

    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.8 * len(dimensions) + 2.2)))

    for offset, evaluator in zip(offsets, evaluators):
        subset = (
            calibration_df[calibration_df["evaluatorId"] == evaluator]
            .set_index("dimensionLabel")
            .reindex(dimensions)
            .dropna(subset=["meanDeviationFromPeers"])
        )
        if subset.empty:
            continue

        positions = np.array([dimensions.index(label) for label in subset.index])
        lower_error = subset["meanDeviationFromPeers"].values - subset["lowerCI"].values
        upper_error = subset["upperCI"].values - subset["meanDeviationFromPeers"].values

        ax.errorbar(
            subset["meanDeviationFromPeers"].values,
            positions + offset,
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            capsize=4,
            markersize=6,
            label=evaluator_display_name(evaluator),
            )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y_base)
    ax.set_yticklabels(dimensions)
    ax.set_xlabel("Mean rating deviation from peer average")
    ax.set_ylabel("Rating dimension")
    ax.set_title("Evaluator Calibration Relative to Peer Ratings")
    ax.legend(title="Evaluator", bbox_to_anchor=(1.02, 1), loc="upper left")
    apply_standard_axes_style(ax, grid_axis="x")

    max_abs = max(
        abs(float(calibration_df["lowerCI"].min())),
        abs(float(calibration_df["upperCI"].max())),
        0.25,
    )
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

    save_figure(
        fig,
        slug,
        "Evaluator Calibration Relative to Peer Ratings",
        "Positive values indicate that an evaluator tended to rate above the average "
        "of the other official evaluators for the same poems; negative values indicate "
        "relatively stricter ratings. Error bars show approximate 95% confidence intervals.",
    )


# ---------------------------------------------------------------------------
# 64 - Supplementary ordinal agreement table
# ---------------------------------------------------------------------------


def export_weighted_kappa_summary(ratings_df: pd.DataFrame) -> None:
    """Export ordinal pairwise agreement metrics for supplementary reporting."""
    kappa_df = _weighted_kappa_summary(ratings_df)
    if not kappa_df.empty:
        save_analysis_table( kappa_df, "pairwise_weighted_kappa_summary", index=False, )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_evaluators(ratings_df: pd.DataFrame | None = None) -> None:
    """Generate evaluator coverage, reliability, agreement, and calibration outputs."""
    if ratings_df is None:
        ratings_df = _load_ratings()

    ratings_df = _prepare_ratings(ratings_df)
    if ratings_df.empty:
        return

    plot_icc_confidence_intervals(ratings_df)
    plot_pairwise_icc_heatmaps(ratings_df)
    plot_evaluator_calibration(ratings_df)
    export_weighted_kappa_summary(ratings_df)
