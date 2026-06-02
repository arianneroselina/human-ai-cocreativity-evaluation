import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import TABLE_DIR, PARTICIPANT_LIKERT_COLUMNS
from scripts.dashboard_figures.utils import save_figure


def plot_participant_age_distribution(participant_df):
    slug = "41_participant_age_distribution"

    if participant_df.empty or "age" not in participant_df.columns:
        return

    age = pd.to_numeric(participant_df["age"], errors="coerce").dropna()

    if age.empty:
        return

    age.describe().to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    bins = range(
        int(age.min()) - 1,
        int(age.max()) + 2,
        2,
        )

    ax.hist(age, bins=bins)

    mean_age = age.mean()
    median_age = age.median()

    ax.set_title("Participant Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of participants")

    summary_text = (
        f"n = {len(age)}\n"
        f"Mean = {mean_age:.1f}\n"
        f"Median = {median_age:.1f}"
    )

    ax.text(
        0.98,
        0.95,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.9,
        },
    )

    save_figure(
        fig,
        slug,
        "Participant Age Distribution",
        "Age distribution of study participants grouped into age intervals.",
    )


def plot_participant_category_distribution(participant_df, column, label, slug):
    if participant_df.empty or column not in participant_df.columns:
        return

    counts = participant_df[column].dropna().value_counts()

    if counts.empty:
        return

    percentages = (counts / counts.sum()) * 100

    export_df = pd.DataFrame({
        "count": counts,
        "percentage": percentages,
    })

    export_df.to_csv(TABLE_DIR / f"{slug}.csv")

    fig_height = max(4.2, 0.35 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    percentages.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(label)
    ax.set_xlabel("Participants (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)

    for container in ax.containers:
        labels = [
            f"{value:.1f}%"
            for value in container.datavalues
        ]

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9,
        )

    save_figure(
        fig,
        slug,
        label,
        f"Participant distribution by {label.lower()} shown as percentages.",
    )


def plot_participant_likert_means(participant_df):
    slug = "46_participant_ai_attitude_means"

    if participant_df.empty:
        return

    rows = []

    for column, label in PARTICIPANT_LIKERT_COLUMNS.items():
        if column not in participant_df.columns:
            continue

        values = pd.to_numeric(participant_df[column], errors="coerce").dropna()

        if values.empty:
            continue

        rows.append({
            "measure": label,
            "mean": values.mean(),
            "n": len(values),
        })

    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    plot_df = summary_df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(plot_df["measure"], plot_df["mean"])

    ax.set_title("Participant Writing Confidence and AI Attitudes")
    ax.set_xlabel("Mean rating (1-5)")
    ax.set_ylabel("")
    ax.set_xlim(1, 5)

    save_figure(
        fig,
        slug,
        "Participant Writing Confidence and AI Attitudes",
        "Mean ratings for writing confidence and attitudes toward AI.",
    )


def plot_participant_info(participant_df):
    if participant_df.empty:
        return

    plot_participant_age_distribution(participant_df)

    plot_participant_category_distribution(
        participant_df,
        "gender",
        "Participant Gender Distribution",
        "42_participant_gender_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "education",
        "Participant Education Distribution",
        "43_participant_education_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "nativeLanguage",
        "Participant Native Language Distribution",
        "44_participant_native_language_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "englishLevel",
        "Participant English Level Distribution",
        "45_participant_english_level_distribution",
    )

    plot_participant_likert_means(participant_df)
