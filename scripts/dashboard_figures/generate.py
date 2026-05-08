import matplotlib

from scripts.config import FIGURE_DIR, TABLE_DIR
from scripts.dashboard_figures.plots_summary import generate_workflow_feedback_summaries

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .loaders import load_master_dataset, load_final_feedback, load_participant_info
from .plots_core import (
    plot_workflow_distribution,
    plot_workflow_transitions,
    plot_mean_quality_by_workflow,
    plot_subjective_feedback_by_workflow,
    plot_ai_performance_over_rounds,
    plot_constraint_rate_by_workflow,
    plot_quality_vs_time,
    plot_final_workflow_ranking,
)
from .plots_participants import plot_participant_info
from .utils import MANIFEST, save_manifest


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_master_dataset()
    feedback_df = load_final_feedback()
    participant_info_df = load_participant_info()

    plot_workflow_distribution(df)
    plot_workflow_transitions(df)
    plot_mean_quality_by_workflow(df)
    plot_subjective_feedback_by_workflow(df)
    plot_ai_performance_over_rounds(df)
    plot_constraint_rate_by_workflow(df)
    plot_quality_vs_time(df)
    plot_final_workflow_ranking(feedback_df)

    plot_participant_info(participant_info_df)
    generate_workflow_feedback_summaries(df, feedback_df)

    save_manifest()

    print(f"Generated {len(MANIFEST)} figures.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
