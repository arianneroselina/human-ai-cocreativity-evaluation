"""Mixed-effects models for poem overall-quality ratings.

Primary outcome
---------------
meanOverallQuality: the mean 1–5 overall-quality score across the three
official evaluators.

Models
------
1. Practice rounds:
   meanOverallQuality ~ round + workflow + participant random intercept

2. Main rounds:
   meanOverallQuality ~ round + workflow + participant random intercept

Practice and Main rounds are modelled separately because workflow assignment
differs by phase: practice workflows were assigned, while Main-round workflows
were chosen voluntarily. Main-round workflow coefficients are therefore
descriptive associations, not causal workflow effects.

Outputs
-------
analysis/quality_model_cell_counts.csv
analysis/quality_model_fixed_effects.csv
analysis/quality_model_overview.csv
analysis/quality_model_practice_round_workflow.txt
analysis/quality_model_main_round_workflow.txt
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.formula.api import mixedlm

from scripts.config import (
    ANALYSIS_DIR,
    MAIN_ROUND_INDICES,
    MASTER_DATASET_PATH,
    PRACTICE_ROUND_INDICES,
    QUALITY_PRIMARY_METRIC,
    WORKFLOW_ORDER,
)
from scripts.dashboard_figures.helpers import drop_duplicate_participant_rounds
from scripts.utils import require_columns


@dataclass
class FittedModel:
    """Store a fitted model and the metadata needed for export."""

    phase: str
    formula: str
    result: object
    data: pd.DataFrame
    fit_method: str
    warnings: list[str]


def load_master_dataset() -> pd.DataFrame:
    """Load the round-level master dataset containing quality metadata."""
    if not MASTER_DATASET_PATH.exists():
        raise FileNotFoundError(f"Master dataset not found: {MASTER_DATASET_PATH}")

    return pd.read_csv(MASTER_DATASET_PATH)


def _assign_phase(df: pd.DataFrame) -> pd.Series:
    """Return a normalized Practice/Main phase label for every round."""
    if "phase" in df.columns:
        phase = df["phase"].astype("string").str.strip().str.lower()
        if phase.isin(["practice", "main"]).all():
            return phase

    return pd.Series(
        np.select(
            [
                df["roundIndex"].isin(PRACTICE_ROUND_INDICES),
                df["roundIndex"].isin(MAIN_ROUND_INDICES),
            ],
            ["practice", "main"],
            default=pd.NA,
        ),
        index=df.index,
        dtype="string",
    )


def prepare_quality_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one valid overall-quality observation per participant and round."""
    required_columns = {
        "participantId",
        "roundIndex",
        "workflow",
        QUALITY_PRIMARY_METRIC,
    }

    if df.empty or not require_columns(
        df,
        required_columns,
        "quality mixed-effects analysis",
    ):
        return pd.DataFrame()

    prepared = drop_duplicate_participant_rounds(df.copy())
    prepared["participantId"] = prepared["participantId"].astype(str)
    prepared["roundIndex"] = pd.to_numeric(
        prepared["roundIndex"],
        errors="coerce",
    )
    prepared[QUALITY_PRIMARY_METRIC] = pd.to_numeric(
        prepared[QUALITY_PRIMARY_METRIC],
        errors="coerce",
    )
    prepared = prepared.dropna(
        subset=[
            "participantId",
            "roundIndex",
            "workflow",
            QUALITY_PRIMARY_METRIC,
        ]
    )
    prepared["roundIndex"] = prepared["roundIndex"].astype(int)
    prepared = prepared[prepared["workflow"].isin(WORKFLOW_ORDER)].copy()

    prepared["phase"] = _assign_phase(prepared)
    prepared = prepared[prepared["phase"].isin(["practice", "main"])].copy()

    return prepared


def build_cell_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observations and participants in every model cell."""
    return (
        data.groupby(["phase", "roundIndex", "workflow"], observed=False)
        .agg(
            outputs=(QUALITY_PRIMARY_METRIC, "size"),
            participants=("participantId", "nunique"),
            meanOverallQuality=(QUALITY_PRIMARY_METRIC, "mean"),
            sdOverallQuality=(QUALITY_PRIMARY_METRIC, "std"),
        )
        .reset_index()
        .sort_values(["phase", "roundIndex", "workflow"])
    )


def _phase_data(data: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Return one model-ready study phase with stable reference categories."""
    phase_df = data[data["phase"].eq(phase)].copy()

    round_order = PRACTICE_ROUND_INDICES if phase == "practice" else MAIN_ROUND_INDICES
    available_rounds = [
        round_index
        for round_index in round_order
        if round_index in set(phase_df["roundIndex"])
    ]

    phase_df = phase_df[phase_df["roundIndex"].isin(available_rounds)].copy()
    phase_df["roundIndex"] = pd.Categorical(
        phase_df["roundIndex"],
        categories=available_rounds,
        ordered=True,
    )
    phase_df["workflow"] = pd.Categorical(
        phase_df["workflow"],
        categories=WORKFLOW_ORDER,
        ordered=True,
    )

    return phase_df


def _model_formula(phase_df: pd.DataFrame) -> str:
    """Build a main-effects formula with the first available round as reference."""
    rounds = list(phase_df["roundIndex"].cat.categories)

    if not rounds:
        raise ValueError("No rounds available for the phase model.")

    first_round = int(rounds[0])

    return (
        f"{QUALITY_PRIMARY_METRIC} ~ "
        f"C(roundIndex, Treatment(reference={first_round})) + "
        "C(workflow, Treatment(reference='human'))"
    )


def fit_phase_model(data: pd.DataFrame, phase: str) -> FittedModel | None:
    """Fit one random-intercept model for Practice or Main rounds."""
    phase_df = _phase_data(data, phase)

    if phase_df.empty:
        print(f"Skipping {phase} model: no valid quality observations.")
        return None

    if phase_df["participantId"].nunique() < 2:
        print(f"Skipping {phase} model: fewer than two participants.")
        return None

    formula = _model_formula(phase_df)
    model = mixedlm(
        formula,
        data=phase_df,
        groups=phase_df["participantId"],
    )

    errors = []

    for method in ["lbfgs", "powell"]:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            try:
                result = model.fit(
                    reml=True,
                    method=method,
                    maxiter=2_000,
                    disp=False,
                )
            except Exception as error:
                errors.append(f"{method}: {error}")
                continue

        warning_messages = [str(warning.message) for warning in caught_warnings]

        if getattr(result, "converged", False):
            return FittedModel(
                phase=phase,
                formula=formula,
                result=result,
                data=phase_df,
                fit_method=method,
                warnings=warning_messages,
            )

        errors.append(
            f"{method}: model did not converge"
            + (f" ({'; '.join(warning_messages)})" if warning_messages else "")
        )

    print(
        f"Skipping {phase} model: unable to obtain a converged fit. "
        + " | ".join(errors)
    )
    return None


def fixed_effects_table(fitted_model: FittedModel) -> pd.DataFrame:
    """Return estimated fixed effects with approximate 95% confidence intervals."""
    result = fitted_model.result
    estimates = result.fe_params
    standard_errors = result.bse_fe
    p_values = result.pvalues.reindex(estimates.index)

    output = pd.DataFrame(
        {
            "model": f"{fitted_model.phase}_round_workflow",
            "phase": fitted_model.phase,
            "term": estimates.index,
            "estimate": estimates.values,
            "standardError": standard_errors.reindex(estimates.index).values,
            "zValue": (estimates / standard_errors.reindex(estimates.index)).values,
            "pValue": p_values.values,
        }
    )
    output["lowerCI95"] = output["estimate"] - 1.96 * output["standardError"]
    output["upperCI95"] = output["estimate"] + 1.96 * output["standardError"]
    output["referenceWorkflow"] = "human"
    output["referenceRound"] = int(fitted_model.data["roundIndex"].cat.categories[0])
    output["outcome"] = QUALITY_PRIMARY_METRIC
    output["observations"] = int(len(fitted_model.data))
    output["participants"] = int(fitted_model.data["participantId"].nunique())

    return output


def model_overview_row(fitted_model: FittedModel) -> dict:
    """Return one compact overview row for a fitted mixed-effects model."""
    result = fitted_model.result

    random_intercept_variance = np.nan
    if getattr(result, "cov_re", None) is not None and not result.cov_re.empty:
        random_intercept_variance = float(result.cov_re.iloc[0, 0])

    return {
        "model": f"{fitted_model.phase}_round_workflow",
        "phase": fitted_model.phase,
        "outcome": QUALITY_PRIMARY_METRIC,
        "formula": fitted_model.formula,
        "observations": int(len(fitted_model.data)),
        "participants": int(fitted_model.data["participantId"].nunique()),
        "rounds": ", ".join(
            str(round_index)
            for round_index in fitted_model.data["roundIndex"].cat.categories
        ),
        "fitMethod": fitted_model.fit_method,
        "converged": bool(fitted_model.result.converged),
        "logLikelihood": float(fitted_model.result.llf),
        "aic": float(fitted_model.result.aic),
        "bic": float(fitted_model.result.bic),
        "residualVariance": float(fitted_model.result.scale),
        "randomInterceptVariance": random_intercept_variance,
        "fitWarnings": " | ".join(fitted_model.warnings),
    }


def write_model_report(fitted_model: FittedModel) -> Path:
    """Write a readable model summary for one phase."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        ANALYSIS_DIR / f"quality_model_{fitted_model.phase}_round_workflow.txt"
    )

    reference_round = int(fitted_model.data["roundIndex"].cat.categories[0])

    report = [
        f"Quality mixed-effects model: {fitted_model.phase.title()} rounds",
        "",
        f"Outcome: {QUALITY_PRIMARY_METRIC}",
        "Model: quality ~ categorical round + categorical workflow",
        "Random effect: participant-specific intercept",
        "Workflow reference: human",
        f"Round reference: {reference_round}",
        f"Observations: {len(fitted_model.data)}",
        f"Participants: {fitted_model.data['participantId'].nunique()}",
        f"Fit method: {fitted_model.fit_method}",
        f"Converged: {fitted_model.result.converged}",
        "",
        "Interpretation note:",
        (
            "Practice workflows were assigned. Main-round workflows were "
            "self-selected, so Main-round workflow estimates are descriptive "
            "associations rather than causal effects."
        ),
        "",
    ]

    if fitted_model.warnings:
        report.extend(
            [
                "Fit warnings:",
                *[f"- {warning}" for warning in fitted_model.warnings],
                "",
            ]
        )

    report.extend(
        [
            "Statsmodels summary",
            "=" * 72,
            str(fitted_model.result.summary()),
            "",
        ]
    )

    output_path.write_text("\n".join(report), encoding="utf-8")
    return output_path


def main() -> None:
    """Fit separate Practice and Main overall-quality mixed-effects models."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    master_df = load_master_dataset()
    analysis_df = prepare_quality_model_data(master_df)

    if analysis_df.empty:
        print("No valid quality observations available for modelling.")
        return

    print(
        f"Prepared {len(analysis_df)} quality observations from "
        f"{analysis_df['participantId'].nunique()} participants."
    )

    cell_counts = build_cell_counts(analysis_df)
    cell_counts_path = ANALYSIS_DIR / "quality_model_cell_counts.csv"
    cell_counts.to_csv(cell_counts_path, index=False)

    fitted_models = [
        fitted_model
        for phase in ["practice", "main"]
        if (fitted_model := fit_phase_model(analysis_df, phase)) is not None
    ]

    if not fitted_models:
        print("No model converged; no model outputs were written.")
        return

    fixed_effects = pd.concat(
        [fixed_effects_table(model) for model in fitted_models],
        ignore_index=True,
    )
    fixed_effects_path = ANALYSIS_DIR / "quality_model_fixed_effects.csv"
    fixed_effects.to_csv(fixed_effects_path, index=False)

    overview = pd.DataFrame([model_overview_row(model) for model in fitted_models])
    overview_path = ANALYSIS_DIR / "quality_model_overview.csv"
    overview.to_csv(overview_path, index=False)

    report_paths = [write_model_report(model) for model in fitted_models]

    print("\nSaved analysis files:")
    print(f"- {cell_counts_path.name}")
    print(f"- {fixed_effects_path.name}")
    print(f"- {overview_path.name}")
    for report_path in report_paths:
        print(f"- {report_path.name}")


if __name__ == "__main__":
    main()
