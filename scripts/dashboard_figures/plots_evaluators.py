"""Overall-quality evaluator agreement analysis.

Figures
-------
61  Overall-quality rating distribution by evaluator
62  Ordinal Krippendorff's alpha across all evaluators
63  ICC reliability for one evaluator and the evaluator mean
64  Pairwise weighted agreement between evaluators
65  Magnitude of evaluator disagreement
66  Relative evaluator rating tendency
67  Pairwise raw-rating matrices

Supplementary CSV files are exported to the dashboard's Statistical Analysis
section via ``save_analysis_table``.
"""

from __future__ import annotations

import re
from itertools import combinations

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

from scripts.config import (
    EVALUATOR_ORDER,
    RATINGS_EXPORT_PATH,
    EVALUATOR_COLORS,
    OVERALL_QUALITY_COLUMN,
    RATING_SCALE,
)
from scripts.dashboard_figures.helpers import evaluator_display_name
from scripts.dashboard_figures.style import apply_standard_axes_style
from scripts.utils import (
    require_columns,
    save_figure,
    save_table,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _ordered_evaluators(ratings_df: pd.DataFrame) -> list[str]:
    """Return configured evaluator IDs first, followed by unexpected IDs."""
    available = set(ratings_df["evaluatorId"].dropna().astype(str))
    ordered = [evaluator for evaluator in EVALUATOR_ORDER if evaluator in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def _load_ratings() -> pd.DataFrame:
    """Load the raw evaluator export used by the dashboard."""
    if not RATINGS_EXPORT_PATH.exists():
        print(
            "Skipping evaluator figures; ratings export not found: "
            f"{RATINGS_EXPORT_PATH}"
        )
        return pd.DataFrame()
    return pd.read_csv(RATINGS_EXPORT_PATH)


def _prepare_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw 1-5 overall-quality ratings."""
    required = {"poemId", "evaluatorId", OVERALL_QUALITY_COLUMN}
    if ratings_df.empty or not require_columns(
        ratings_df,
        required,
        "overall-quality evaluator ratings",
    ):
        return pd.DataFrame()

    prepared = ratings_df[["poemId", "evaluatorId", OVERALL_QUALITY_COLUMN]].copy()
    prepared["evaluatorId"] = prepared["evaluatorId"].astype(str)
    prepared[OVERALL_QUALITY_COLUMN] = pd.to_numeric(
        prepared[OVERALL_QUALITY_COLUMN],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=["poemId", "evaluatorId", OVERALL_QUALITY_COLUMN])

    invalid = ~prepared[OVERALL_QUALITY_COLUMN].isin(RATING_SCALE)
    if invalid.any():
        print(
            "Dropping ratings outside the expected 1-5 scale: "
            f"{int(invalid.sum())} row(s)."
        )
        prepared = prepared.loc[~invalid].copy()

    return prepared.drop_duplicates(
        subset=["poemId", "evaluatorId"],
        keep="last",
    )


def _complete_rating_panel(
    ratings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], int]:
    """Return a complete poem × evaluator rating matrix."""
    evaluators = _ordered_evaluators(ratings_df)
    wide_df = ratings_df.pivot(
        index="poemId",
        columns="evaluatorId",
        values=OVERALL_QUALITY_COLUMN,
    ).reindex(columns=evaluators)

    total_poems = int(len(wide_df))
    complete_df = wide_df.dropna(how="any").copy()
    return complete_df, evaluators, total_poems


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _parse_ci(value) -> tuple[float, float] | None:
    """Parse Pingouin confidence intervals from supported package versions."""
    if value is None:
        return None

    if isinstance(value, str):
        numbers = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        return None

    try:
        return float(value[0]), float(value[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _select_icc_row(
    icc_table: pd.DataFrame,
    aliases: list[str],
) -> pd.Series | None:
    """Find one ICC row despite Pingouin naming differences."""
    for alias in aliases:
        match = icc_table[icc_table["Type"].eq(alias)]
        if not match.empty:
            return match.iloc[0]
    return None


def _long_panel(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a complete rating matrix to Pingouin's long format."""
    return (
        wide_df.rename_axis(index="poemId", columns="evaluatorId")
        .stack()
        .rename("rating")
        .reset_index()
    )


def _icc_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate absolute-agreement ICC(A,1) and ICC(A,k)."""
    if wide_df.empty or wide_df.shape[0] < 2 or wide_df.shape[1] < 2:
        return pd.DataFrame()

    try:
        icc_table = pg.intraclass_corr(
            data=_long_panel(wide_df),
            targets="poemId",
            raters="evaluatorId",
            ratings="rating",
        )
    except Exception as error:
        print(f"Unable to calculate overall-quality ICC: {error}")
        return pd.DataFrame()

    requested = {
        "ICC(A,1)": ["ICC(A,1)", "ICC2"],
        "ICC(A,k)": ["ICC(A,k)", "ICC2k"],
    }
    rows = []

    for statistic, aliases in requested.items():
        row = _select_icc_row(icc_table, aliases)
        if row is None:
            continue

        ci = _parse_ci(row.get("CI95%", row.get("CI95", None)))
        if ci is None:
            continue

        rows.append(
            {
                "statistic": statistic,
                "interpretation": (
                    "Reliability of one individual evaluator"
                    if statistic == "ICC(A,1)"
                    else "Reliability of the mean across the full evaluator panel"
                ),
                "icc": float(row["ICC"]),
                "lowerCI": ci[0],
                "upperCI": ci[1],
                "fValue": float(row.get("F", np.nan)),
                "pValue": float(row.get("pval", np.nan)),
                "nPoems": int(wide_df.shape[0]),
                "nEvaluators": int(wide_df.shape[1]),
            }
        )

    return pd.DataFrame(rows)


def _pairwise_summary(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> pd.DataFrame:
    """Calculate direct pairwise absolute and ordinal agreement metrics."""
    rows = []

    for evaluator_a, evaluator_b in combinations(evaluators, 2):
        ratings_a = wide_df[evaluator_a].astype(int)
        ratings_b = wide_df[evaluator_b].astype(int)
        differences = ratings_a - ratings_b

        pair_wide = wide_df[[evaluator_a, evaluator_b]].copy()
        pair_wide.columns = ["A", "B"]
        pair_long = (
            pair_wide.rename_axis(index="poemId", columns="evaluatorId")
            .stack()
            .rename("rating")
            .reset_index()
        )

        pair_icc = np.nan
        try:
            pair_icc_table = pg.intraclass_corr(
                data=pair_long,
                targets="poemId",
                raters="evaluatorId",
                ratings="rating",
            )
            pair_icc_row = _select_icc_row(pair_icc_table, ["ICC(A,1)", "ICC2"])
            if pair_icc_row is not None:
                pair_icc = float(pair_icc_row["ICC"])
        except Exception:
            pass

        rows.append(
            {
                "evaluatorA": evaluator_a,
                "evaluatorALabel": evaluator_display_name(evaluator_a),
                "evaluatorB": evaluator_b,
                "evaluatorBLabel": evaluator_display_name(evaluator_b),
                "pairLabel": (
                    f"{evaluator_display_name(evaluator_a)} vs "
                    f"{evaluator_display_name(evaluator_b)}"
                ),
                "pairwiseIccA1": pair_icc,
                "quadraticWeightedKappa": cohen_kappa_score(
                    ratings_a,
                    ratings_b,
                    weights="quadratic",
                ),
                "exactAgreementPercentage": float((differences == 0).mean() * 100),
                "withinOnePointPercentage": float(
                    (differences.abs() <= 1).mean() * 100
                ),
                "meanDifferenceAminusB": float(differences.mean()),
                "meanAbsoluteDifference": float(differences.abs().mean()),
                "spearmanRho": float(ratings_a.corr(ratings_b, method="spearman")),
                "nSharedPoems": int(len(wide_df)),
            }
        )

    return pd.DataFrame(rows)


def _rating_distribution(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> pd.DataFrame:
    """Return count and percentage by evaluator and raw 1-5 rating."""
    rows = []
    total_poems = len(wide_df)

    for evaluator in evaluators:
        counts = (
            wide_df[evaluator]
            .value_counts()
            .reindex(
                RATING_SCALE,
                fill_value=0,
            )
        )
        for rating, count in counts.items():
            rows.append(
                {
                    "evaluatorId": evaluator,
                    "evaluatorLabel": evaluator_display_name(evaluator),
                    "rating": int(rating),
                    "poemCount": int(count),
                    "percentage": float(count / total_poems * 100),
                    "nPoems": int(total_poems),
                }
            )

    return pd.DataFrame(rows)


def _tendency_summary(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> pd.DataFrame:
    """Compare each evaluator with the average rating of the other two."""
    rows = []

    for evaluator in evaluators:
        peers = [peer for peer in evaluators if peer != evaluator]
        deviations = wide_df[evaluator] - wide_df[peers].mean(axis=1)
        count = len(deviations)
        standard_error = (
            deviations.std(ddof=1) / np.sqrt(count) if count > 1 else np.nan
        )
        margin = 1.96 * standard_error if np.isfinite(standard_error) else np.nan

        rows.append(
            {
                "evaluatorId": evaluator,
                "evaluatorLabel": evaluator_display_name(evaluator),
                "meanRating": float(wide_df[evaluator].mean()),
                "meanDeviationFromPeers": float(deviations.mean()),
                "lowerCI": float(deviations.mean() - margin)
                if np.isfinite(margin)
                else np.nan,
                "upperCI": float(deviations.mean() + margin)
                if np.isfinite(margin)
                else np.nan,
                "nPoems": int(count),
            }
        )

    return pd.DataFrame(rows)


def _disagreement_outputs(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return distribution of evaluator ranges and poem-level disagreement rows."""
    disagreements = wide_df.copy()
    disagreements["meanOverallQuality"] = disagreements[evaluators].mean(axis=1)
    disagreements["ratingRange"] = (
        disagreements[evaluators].max(axis=1) - disagreements[evaluators].min(axis=1)
    ).astype(int)

    distribution = (
        disagreements["ratingRange"]
        .value_counts()
        .reindex(range(0, 5), fill_value=0)
        .rename_axis("ratingRange")
        .reset_index(name="poemCount")
    )
    distribution["percentage"] = distribution["poemCount"] / len(disagreements) * 100
    distribution["interpretation"] = distribution["ratingRange"].map(
        {
            0: "All three evaluators gave the same rating",
            1: "All ratings were within one point",
            2: "At least one two-point disagreement",
            3: "At least one three-point disagreement",
            4: "Full-scale disagreement",
        }
    )

    rename_map = {
        evaluator: evaluator_display_name(evaluator) for evaluator in evaluators
    }
    detailed = (
        disagreements.reset_index()
        .rename(columns=rename_map)
        .sort_values(["ratingRange", "meanOverallQuality"], ascending=[False, True])
    )

    return distribution, detailed


def _ordinal_krippendorff_alpha_summary(
    wide_df: pd.DataFrame,
    evaluators: list[str],
    *,
    bootstrap_iterations: int = 5000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Calculate ordinal Krippendorff's alpha and a poem-level bootstrap CI.

    The reliability matrix must have evaluators in rows and poems in columns.
    Missing ratings may be represented by NaN.
    """
    ratings = (
        wide_df[evaluators]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        .T
    )

    if ratings.shape[0] < 2 or ratings.shape[1] < 2:
        return pd.DataFrame()

    alpha = float(
        krippendorff.alpha(
            reliability_data=ratings,
            level_of_measurement="ordinal",
        )
    )

    rng = np.random.default_rng(random_seed)
    poem_count = ratings.shape[1]
    bootstrap_values = []

    # Resample complete poems, preserving the three ratings belonging
    # to each poem.
    for _ in range(bootstrap_iterations):
        sampled_columns = rng.integers(
            low=0,
            high=poem_count,
            size=poem_count,
        )
        sampled_ratings = ratings[:, sampled_columns]

        try:
            bootstrap_alpha = krippendorff.alpha(
                reliability_data=sampled_ratings,
                level_of_measurement="ordinal",
            )
        except (ValueError, ZeroDivisionError):
            continue

        if np.isfinite(bootstrap_alpha):
            bootstrap_values.append(float(bootstrap_alpha))

    if bootstrap_values:
        lower_ci, upper_ci = np.quantile(
            bootstrap_values,
            [0.025, 0.975],
        )
    else:
        lower_ci = np.nan
        upper_ci = np.nan

    return pd.DataFrame(
        [
            {
                "statistic": "Ordinal Krippendorff's alpha",
                "alpha": alpha,
                "lowerCI": lower_ci,
                "upperCI": upper_ci,
                "poems": poem_count,
                "evaluators": len(evaluators),
                "bootstrapIterations": len(bootstrap_values),
            }
        ]
    )


# ---------------------------------------------------------------------------
# 61: Raw rating distribution
# ---------------------------------------------------------------------------


def plot_overall_quality_rating_distribution(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Show the raw 1-5 overall-quality distribution of each evaluator."""
    slug = "61_overall_quality_rating_distribution"
    distribution_df = _rating_distribution(wide_df, evaluators)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(RATING_SCALE))
    width = 0.76 / len(evaluators)
    offsets = np.linspace(
        -(len(evaluators) - 1) * width / 2,
        (len(evaluators) - 1) * width / 2,
        len(evaluators),
    )

    for offset, evaluator in zip(offsets, evaluators):
        subset = (
            distribution_df[distribution_df["evaluatorId"].eq(evaluator)]
            .set_index("rating")
            .reindex(RATING_SCALE)
        )
        bars = ax.bar(
            positions + offset,
            subset["percentage"].to_numpy(dtype=float),
            width=width,
            color=EVALUATOR_COLORS[evaluator],
            edgecolor="white",
            linewidth=0.8,
            label=evaluator_display_name(evaluator),
            zorder=2,
        )

        for bar, count in zip(bars, subset["poemCount"].to_numpy(dtype=int)):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.1,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(positions)
    ax.set_xticklabels(RATING_SCALE)
    ax.set_ylim(0, max(5, float(distribution_df["percentage"].max()) + 10))
    ax.set_xlabel("Raw overall-quality rating")
    ax.set_ylabel("Rated poems (%)")
    ax.set_title("Overall-Quality Rating Distribution by Evaluator")
    ax.legend(title="Official evaluator")
    apply_standard_axes_style(ax, grid_axis="y")
    fig.tight_layout()

    save_figure(
        fig,
        slug,
        "Overall-Quality Rating Distribution by Evaluator",
        "Raw 1-5 overall-quality ratings assigned to the same complete set of "
        "poems by each official evaluator. Labels show poem counts.",
    )


# ---------------------------------------------------------------------------
# 62: Ordinal Krippendorff's alpha
# ---------------------------------------------------------------------------


def plot_overall_quality_ordinal_agreement(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Plot ordinal Krippendorff's alpha across the full evaluator panel."""
    slug = "62_overall_quality_ordinal_agreement"

    alpha_df = _ordinal_krippendorff_alpha_summary(
        wide_df,
        evaluators,
    )
    if alpha_df.empty:
        return

    save_table(
        alpha_df,
        f"{slug}_summary",
        index=False,
    )

    alpha_row = alpha_df.iloc[0]
    alpha = float(alpha_row["alpha"])
    lower_ci = float(alpha_row["lowerCI"])
    upper_ci = float(alpha_row["upperCI"])

    fig, ax = plt.subplots(figsize=(7.4, 3.8))

    if np.isfinite(lower_ci) and np.isfinite(upper_ci):
        lower_error = max(alpha - lower_ci, 0.0)
        upper_error = max(upper_ci - alpha, 0.0)

        ax.errorbar(
            [alpha],
            [0],
            xerr=np.array([[lower_error], [upper_error]]),
            fmt="o",
            color="#333333",
            markersize=10,
            capsize=5,
            linewidth=1.8,
            zorder=3,
        )
        ci_label = f"95% CI [{lower_ci:.2f}, {upper_ci:.2f}]"
    else:
        ax.scatter(
            [alpha],
            [0],
            s=100,
            color="#333333",
            zorder=3,
        )
        ci_label = "95% CI unavailable"

    ax.annotate(
        f"αordinal = {alpha:.3f}\n{ci_label}",
        (alpha, 0),
        xytext=(12, 0),
        textcoords="offset points",
        va="center",
        fontsize=10,
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlim(-0.05, 1.02)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([0])
    ax.set_yticklabels([f"All {len(evaluators)} evaluators"])
    ax.set_xlabel("Ordinal Krippendorff's alpha")
    ax.set_title("Overall-Quality Ordinal Inter-Rater Agreement")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            f"Agreement across {len(wide_df)} poems rated by "
            f"{len(evaluators)} evaluators. The confidence interval was estimated "
            "by resampling poems."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Overall-Quality Ordinal Inter-Rater Agreement",
        (
            "Ordinal Krippendorff's alpha and bootstrap 95% confidence interval "
            "across all official evaluators."
        ),
    )


# ---------------------------------------------------------------------------
# 63: ICC reliability
# ---------------------------------------------------------------------------


def plot_overall_quality_icc_reliability(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Plot absolute-agreement ICC for one evaluator and the evaluator mean."""
    slug = "63_overall_quality_icc_reliability"

    icc_df = _icc_summary(wide_df)
    if icc_df.empty:
        return

    save_table(
        icc_df,
        f"{slug}_summary",
        index=False,
    )

    icc_plot = (
        icc_df.set_index("statistic")
        .reindex(["ICC(A,1)", "ICC(A,k)"])
        .dropna(subset=["icc"])
        .reset_index()
    )
    if icc_plot.empty:
        return

    y_positions = np.arange(len(icc_plot))
    values = icc_plot["icc"].to_numpy(dtype=float)
    lower_ci = icc_plot["lowerCI"].to_numpy(dtype=float)
    upper_ci = icc_plot["upperCI"].to_numpy(dtype=float)

    lower_error = np.maximum(values - lower_ci, 0.0)
    upper_error = np.maximum(upper_ci - values, 0.0)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))

    ax.errorbar(
        values,
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="o",
        color="#333333",
        markersize=9,
        capsize=5,
        linewidth=1.7,
        zorder=3,
    )

    for y_position, (_, row) in zip(y_positions, icc_plot.iterrows()):
        ax.annotate(
            (f"{row['icc']:.3f}\n95% CI [{row['lowerCI']:.2f}, {row['upperCI']:.2f}]"),
            (row["icc"], y_position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlim(-0.05, 1.02)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            "One evaluator\nICC(A,1)",
            f"Mean of {len(evaluators)} evaluators\nICC(A,k)",
        ]
    )
    ax.set_xlabel("Absolute-agreement intraclass correlation")
    ax.set_title("Overall-Quality ICC Reliability")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            f"ICC(A,1) describes one evaluator's score; ICC(A,k) describes the "
            f"mean score across {len(evaluators)} evaluators for {len(wide_df)} poems."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Overall-Quality ICC Reliability",
        (
            "Absolute-agreement ICC values and 95% confidence intervals for one "
            "evaluator and the mean across the full evaluator panel."
        ),
    )


# ---------------------------------------------------------------------------
# 64: Pairwise weighted agreement
# ---------------------------------------------------------------------------


def plot_pairwise_overall_quality_agreement(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Plot pairwise weighted kappa, exact agreement, and one-point agreement."""
    slug = "64_pairwise_overall_quality_agreement"

    pairwise_df = _pairwise_summary(
        wide_df,
        evaluators,
    )
    if pairwise_df.empty:
        return

    save_table(
        pairwise_df,
        f"{slug}_summary",
        index=False,
    )

    pairwise_plot = pairwise_df.copy()
    pairwise_plot["containsAI"] = pairwise_plot["evaluatorALabel"].str.contains(
        "AI",
        case=False,
        na=False,
    ) | pairwise_plot["evaluatorBLabel"].str.contains(
        "AI",
        case=False,
        na=False,
    )

    # Human-human comparison first, followed by the two AI-involving pairs.
    pairwise_plot = pairwise_plot.sort_values(
        ["containsAI", "pairLabel"],
        ascending=[True, True],
    ).reset_index(drop=True)

    y_positions = np.arange(len(pairwise_plot))
    bar_colours = [
        "#9a5f53" if contains_ai else "#888888"
        for contains_ai in pairwise_plot["containsAI"]
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.6))

    ax.barh(
        y_positions,
        pairwise_plot["quadraticWeightedKappa"],
        color=bar_colours,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    for y_position, (_, row) in zip(
        y_positions,
        pairwise_plot.iterrows(),
    ):
        ax.text(
            row["quadraticWeightedKappa"] + 0.012,
            y_position,
            (
                f"κw={row['quadraticWeightedKappa']:.3f}  |  "
                f"exact {row['exactAgreementPercentage']:.1f}%  |  "
                f"±1 {row['withinOnePointPercentage']:.1f}%"
            ),
            va="center",
            fontsize=8.4,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlim(-0.03, 1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pairwise_plot["pairLabel"])
    ax.invert_yaxis()
    ax.set_xlabel("Quadratic-weighted Cohen's kappa")
    ax.set_title("Pairwise Overall-Quality Agreement")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            f"Pairwise agreement across {len(wide_df)} shared poems. "
            "Brown bars involve the AI evaluator; the grey bar compares the two humans."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Pairwise Overall-Quality Agreement",
        (
            "Pairwise quadratic-weighted Cohen's kappa, exact agreement, and "
            "within-one-point agreement for every evaluator pair."
        ),
    )


# ---------------------------------------------------------------------------
# 65: Magnitude of evaluator disagreement
# ---------------------------------------------------------------------------


def plot_evaluator_disagreement_magnitude(
        wide_df: pd.DataFrame,
        evaluators: list[str],
) -> None:
    """Show the poem-level spread across evaluator ratings."""
    slug = "65_evaluator_disagreement_magnitude"
    range_df, disagreement_rows = _disagreement_outputs(wide_df, evaluators)

    if range_df.empty:
        return

    save_table(
        range_df,
        f"{slug}_distribution",
        index=False,
    )
    save_table(
        disagreement_rows,
        f"{slug}_poem_level",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    bars = ax.bar(
        range_df["ratingRange"].astype(str),
        range_df["percentage"],
        color="#8BAE66",
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    for bar, (_, row) in zip(bars, range_df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{int(row['poemCount'])}\n({row['percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            )

    ax.set_ylim(0, max(5, float(range_df["percentage"].max()) + 12))
    ax.set_xlabel("Rating range across all evaluators")
    ax.set_ylabel("Poems (%)")
    ax.set_title("Magnitude of Evaluator Disagreement")
    apply_standard_axes_style(ax, grid_axis="y")

    fig.text(
        0.01,
        0.01,
        (
            "Rating range = highest minus lowest raw 1-5 rating for the same poem. "
            "A range of 0 indicates exact agreement."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Magnitude of Evaluator Disagreement",
        (
            "Distribution of the difference between the highest and lowest "
            "overall-quality rating assigned to each poem."
        ),
    )


# ---------------------------------------------------------------------------
# 66: Relative evaluator rating tendency
# ---------------------------------------------------------------------------


def plot_evaluator_rating_tendency(
    wide_df: pd.DataFrame,
    evaluators: list[str],
) -> None:
    """Show systematic evaluator strictness or generosity."""
    slug = "66_evaluator_rating_tendency"
    tendency_df = _tendency_summary(wide_df, evaluators)

    if tendency_df.empty:
        return

    save_table(
        tendency_df,
        f"{slug}_summary",
        index=False,
    )

    tendency_plot = tendency_df.reset_index(drop=True)
    y_positions = np.arange(len(tendency_plot))
    deviations = tendency_plot["meanDeviationFromPeers"].to_numpy(dtype=float)
    lower_error = deviations - tendency_plot["lowerCI"].to_numpy(dtype=float)
    upper_error = tendency_plot["upperCI"].to_numpy(dtype=float) - deviations

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.errorbar(
        deviations,
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="none",
        capsize=4,
        linewidth=1.4,
        color="#333333",
        zorder=2,
    )

    for y_position, (_, row) in zip(y_positions, tendency_plot.iterrows()):
        ax.scatter(
            row["meanDeviationFromPeers"],
            y_position,
            s=70,
            color=EVALUATOR_COLORS[row["evaluatorId"]],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            (
                f"Mean rating {row['meanRating']:.2f}\n"
                f"Peer deviation {row['meanDeviationFromPeers']:+.2f}"
            ),
            (row["meanDeviationFromPeers"], y_position),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
        )

    max_abs = max(
        0.3,
        abs(float(tendency_plot["lowerCI"].min())),
        abs(float(tendency_plot["upperCI"].max())),
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tendency_plot["evaluatorLabel"])
    ax.set_xlabel("Mean deviation from the other evaluators")
    ax.set_ylabel("Evaluator")
    ax.set_title("Relative Evaluator Rating Tendency")
    apply_standard_axes_style(ax, grid_axis="x")

    fig.text(
        0.01,
        0.01,
        (
            "Positive values indicate more generous ratings than the other "
            "evaluators; negative values indicate stricter ratings."
        ),
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_figure(
        fig,
        slug,
        "Relative Evaluator Rating Tendency",
        (
            "Mean deviation of each evaluator's rating from the average rating "
            "of the other evaluators, with 95% confidence intervals."
        ),
    )


# ---------------------------------------------------------------------------
# 67: Direct raw-rating matrices
# ---------------------------------------------------------------------------


def plot_pairwise_overall_quality_matrices(
        wide_df: pd.DataFrame,
        evaluators: list[str],
) -> None:
    """Show every raw 1-5 rating combination for every evaluator pair."""
    slug = "67_pairwise_overall_quality_rating_matrices"
    pairwise_df = _pairwise_summary(wide_df, evaluators)
    if pairwise_df.empty:
        return

    pairs = list(combinations(evaluators, 2))
    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(5.0 * len(pairs), 4.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    images = []

    for axis, (evaluator_a, evaluator_b) in zip(axes_flat, pairs):
        matrix = pd.crosstab(
            wide_df[evaluator_b].astype(int),
            wide_df[evaluator_a].astype(int),
        ).reindex(index=RATING_SCALE, columns=RATING_SCALE, fill_value=0)

        matrix_values = matrix.to_numpy()
        image = axis.imshow(
            matrix_values,
            cmap="Blues",
            vmin=0,
            vmax=max(1, int(matrix_values.max())),
            origin="lower",
            aspect="equal",
        )
        images.append(image)

        for row_index, rating_b in enumerate(RATING_SCALE):
            for column_index, rating_a in enumerate(RATING_SCALE):
                count = int(matrix.loc[rating_b, rating_a])
                text_color = "white" if count > matrix_values.max() * 0.55 else "black"
                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

        pair_metrics = pairwise_df[
            pairwise_df["evaluatorA"].eq(evaluator_a)
            & pairwise_df["evaluatorB"].eq(evaluator_b)
            ].iloc[0]

        axis.set_xticks(range(len(RATING_SCALE)))
        axis.set_xticklabels(RATING_SCALE)
        axis.set_yticks(range(len(RATING_SCALE)))
        axis.set_yticklabels(RATING_SCALE)
        axis.set_xlabel(f"{evaluator_display_name(evaluator_a)} rating")
        axis.set_ylabel(f"{evaluator_display_name(evaluator_b)} rating")
        axis.set_title(
            f"{evaluator_display_name(evaluator_a)} vs "
            f"{evaluator_display_name(evaluator_b)}\n"
            f"κw = {pair_metrics['quadraticWeightedKappa']:.3f}"
        )

    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.16,
        top=0.86,
        wspace=0.35,
    )

    colorbar_axis = fig.add_axes([0.91, 0.17, 0.015, 0.68])

    fig.colorbar(
        images[0],
        cax=colorbar_axis,
        label="Number of poems",
    )

    fig.suptitle("Raw Overall-Quality Rating Combinations", fontsize=13, y=0.99)
    fig.text(
        0.01,
        0.01,
        "Diagonal cells show exact agreement. Cells directly next to the diagonal "
        "represent ratings that differed by one point.",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#4a4a4a",
    )

    save_figure(
        fig,
        slug,
        "Raw Overall-Quality Rating Combinations",
        "Cross-tabulations of raw 1-5 overall-quality ratings for every evaluator "
        "pair. Diagonal cells show exact agreement; off-diagonal cells show the "
        "direction and magnitude of disagreements.",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_evaluators(ratings_df: pd.DataFrame | None = None) -> None:
    """Generate overall-quality evaluator-agreement dashboard outputs."""
    if ratings_df is None:
        ratings_df = _load_ratings()

    prepared = _prepare_ratings(ratings_df)
    if prepared.empty:
        return

    wide_df, evaluators, total_poems = _complete_rating_panel(prepared)
    if wide_df.empty or len(evaluators) < 2:
        print(
            "Skipping evaluator figures; fewer than two evaluators rated a complete "
            "set of overall-quality scores."
        )
        return

    if len(wide_df) != total_poems:
        print(
            "Evaluator figures use only poems with a complete rating panel: "
            f"{len(wide_df)}/{total_poems} poems."
        )

    plot_overall_quality_rating_distribution(wide_df, evaluators)
    plot_overall_quality_ordinal_agreement(wide_df, evaluators)
    plot_overall_quality_icc_reliability(wide_df, evaluators)
    plot_pairwise_overall_quality_agreement(wide_df, evaluators)
    plot_evaluator_disagreement_magnitude(wide_df, evaluators)
    plot_evaluator_rating_tendency(wide_df, evaluators)
    plot_pairwise_overall_quality_matrices(wide_df, evaluators)
