"""Main entry point for dashboard figure generation.

Loads all required datasets, runs each plot module, and writes figures,
tables, and the dashboard manifest.
"""

import matplotlib

from scripts.dashboard_figures.helpers import prepare_round_data
from scripts.dashboard_figures.plots_efficiency import plot_efficiency

# Must be set before importing modules that may import pyplot.
matplotlib.use("Agg")

from scripts.config import FIGURE_DIR, TABLE_DIR
from scripts.dashboard_figures.loaders import (
    load_final_feedback,
    load_master_dataset,
    load_participant_info,
)
from scripts.dashboard_figures.plots_constraints import plot_constraints
from scripts.dashboard_figures.plots_error_exposure import plot_error_exposure
from scripts.dashboard_figures.plots_evaluators import plot_evaluators
from scripts.dashboard_figures.plots_experience import plot_experience
from scripts.dashboard_figures.plots_participants import plot_participant_info
from scripts.dashboard_figures.plots_quality import plot_quality
from scripts.dashboard_figures.plots_summary import generate_feedback_summaries
from scripts.dashboard_figures.plots_workflow import plot_workflow
from scripts.dashboard_figures.style import configure_matplotlib
from scripts.utils import (
    MANIFEST,
    save_manifest,
    reset_manifest,
)


def main() -> None:
    """Generate every figure and table used by the research dashboard."""
    configure_matplotlib()
    reset_manifest()

    round_df = load_master_dataset()
    prepared_round_df = prepare_round_data(round_df)
    feedback_df = load_final_feedback()
    participant_info_df = load_participant_info()

    plot_workflow(prepared_round_df, feedback_df)
    plot_experience(prepared_round_df)
    plot_quality(prepared_round_df)
    plot_efficiency(prepared_round_df)
    plot_constraints(prepared_round_df)
    plot_error_exposure(prepared_round_df, feedback_df)
    plot_evaluators()
    plot_participant_info(participant_info_df)
    generate_feedback_summaries(prepared_round_df, feedback_df)

    save_manifest()

    print(f"Generated {len(MANIFEST)} figures.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
