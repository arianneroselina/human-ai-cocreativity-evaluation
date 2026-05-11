import matplotlib

from scripts.config import FIGURE_DIR, TABLE_DIR
from scripts.dashboard_figures.loaders import load_participant_info, load_final_feedback, load_master_dataset
from scripts.dashboard_figures.plots_error_exposure import plot_error_exposure
from scripts.dashboard_figures.plots_experience import plot_experience
from scripts.dashboard_figures.plots_participants import plot_participant_info
from scripts.dashboard_figures.plots_quality import plot_quality
from scripts.dashboard_figures.plots_summary import generate_feedback_summaries
from scripts.dashboard_figures.plots_workflow import plot_workflow
from scripts.dashboard_figures.utils import save_manifest, MANIFEST

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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

    plot_workflow(df, feedback_df)
    plot_experience(df)
    plot_quality(df)
    plot_error_exposure(df)
    plot_participant_info(participant_info_df)
    generate_feedback_summaries(df, feedback_df)

    save_manifest()

    print(f"Generated {len(MANIFEST)} figures.")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables:  {TABLE_DIR}")


if __name__ == "__main__":
    main()
