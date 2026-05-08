import pandas as pd
from matplotlib import pyplot as plt

from scripts.config import TABLE_DIR, PARTICIPANT_LIKERT_COLUMNS
from scripts.dashboard_figures.utils import save_figure


def plot_participant_age_distribution(participant_df):
    if participant_df.empty or "age" not in participant_df.columns:
        return

    age = participant_df["age"].dropna()

    if age.empty:
        return

    age.describe().to_csv(TABLE_DIR / "participant_age_summary.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(age, bins=min(10, max(3, age.nunique())))

    ax.set_title("Participant Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of participants")

    save_figure(
        fig,
        "10_participant_age_distribution",
        "Participant Age Distribution",
        "Distribution of participant ages.",
    )


def plot_participant_category_distribution(participant_df, column, label, slug):
    if participant_df.empty or column not in participant_df.columns:
        return

    counts = participant_df[column].dropna().value_counts()

    if counts.empty:
        return

    counts.to_csv(TABLE_DIR / f"{slug}.csv", header=["count"])

    fig_height = max(4.2, 0.35 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))

    counts.sort_values().plot(kind="barh", ax=ax)

    ax.set_title(label)
    ax.set_xlabel("Number of participants")
    ax.set_ylabel("")

    save_figure(
        fig,
        slug,
        label,
        f"Participant distribution by {label.lower()}.",
    )


def plot_participant_likert_means(participant_df):
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
    summary_df.to_csv(TABLE_DIR / "participant_ai_attitude_means.csv", index=False)

    plot_df = summary_df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(plot_df["measure"], plot_df["mean"])

    ax.set_title("Participant Writing Confidence and AI Attitudes")
    ax.set_xlabel("Mean rating")
    ax.set_ylabel("")
    ax.set_xlim(1, 5)

    save_figure(
        fig,
        "15_participant_ai_attitude_means",
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
        "11_participant_gender_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "education",
        "Participant Education Distribution",
        "12_participant_education_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "nativeLanguage",
        "Participant Native Language Distribution",
        "13_participant_native_language_distribution",
    )

    plot_participant_category_distribution(
        participant_df,
        "englishLevel",
        "Participant English Level Distribution",
        "14_participant_english_level_distribution",
    )

    plot_participant_likert_means(participant_df)

