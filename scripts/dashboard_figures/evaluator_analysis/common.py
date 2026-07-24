"""Data preparation and agreement summaries for evaluator figures."""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd

from scripts.config import (
    CI_Z_VALUE,
    EVALUATOR_ORDER,
    OVERALL_QUALITY_COLUMN,
    RATING_SCALE,
    RATINGS_EXPORT_PATH,
)
from scripts.dashboard_figures.helpers import evaluator_display_name
from scripts.utils import require_columns


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
        import pingouin as pg
    except ImportError:
        print("Skipping ICC figures; install the optional 'pingouin' package.")
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
    try:
        import pingouin as pg
    except ImportError:
        pg = None
        print("Pairwise ICC unavailable; install the optional 'pingouin' package.")

    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        cohen_kappa_score = None
        print(
            "Weighted kappa unavailable; install the optional 'scikit-learn' package."
        )

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
            if pg is None:
                raise ImportError
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
                "quadraticWeightedKappa": (
                    cohen_kappa_score(
                        ratings_a,
                        ratings_b,
                        weights="quadratic",
                    )
                    if cohen_kappa_score is not None
                    else np.nan
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
        margin = CI_Z_VALUE * standard_error if np.isfinite(standard_error) else np.nan

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
    try:
        import krippendorff
    except ImportError:
        print(
            "Skipping ordinal-agreement figure; install the optional "
            "'krippendorff' package."
        )
        return pd.DataFrame()

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
