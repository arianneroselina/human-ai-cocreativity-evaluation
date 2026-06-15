import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from scripts.config import PARTICIPANT_LIKERT_COLUMNS, TABLE_DIR
from scripts.utils import save_figure


def format_pie_label(values):
    total = sum(values)

    def formatter(percent):
        count = int(round(percent * total / 100))

        if count == 0:
            return ""

        return f"{percent:.1f}%\n(n={count})"

    return formatter


def export_category_distribution(counts, slug):
    percentages = counts / counts.sum() * 100

    export_df = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages.round(2),
        }
    )

    export_df.to_csv(TABLE_DIR / f"{slug}.csv")

    return percentages


def plot_participant_age_distribution(participant_df):
    slug = "41_participant_age_distribution"

    if participant_df.empty or "age" not in participant_df.columns:
        return

    age = pd.to_numeric(participant_df["age"], errors="coerce").dropna()

    if age.empty:
        return

    age.describe().to_csv(TABLE_DIR / f"{slug}.csv")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    bins = range(
        int(age.min()) - 1,
        int(age.max()) + 2,
        2,
    )

    ax.hist(
        age,
        bins=bins,
        edgecolor="white",
        linewidth=1,
    )

    mean_age = age.mean()
    median_age = age.median()

    ax.axvline(
        mean_age,
        color="darkred",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {mean_age:.1f}",
    )

    ax.axvline(
        median_age,
        color="darkgreen",
        linestyle=":",
        linewidth=1.8,
        label=f"Median = {median_age:.1f}",
    )

    ax.set_title("Participant Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of participants")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title=f"n = {len(age)}")

    save_figure(
        fig,
        slug,
        "Participant Age Distribution",
        "Age distribution of study participants with mean and median marked.",
    )


def plot_participant_pie_distribution(participant_df, column, label, slug):
    if participant_df.empty or column not in participant_df.columns:
        return

    counts = participant_df[column].dropna().value_counts()

    if counts.empty:
        return

    percentages = export_category_distribution(counts, slug)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    wedges, _, _ = ax.pie(
        counts.values,
        autopct=format_pie_label(counts.values),
        startangle=90,
        counterclock=False,
        pctdistance=0.72,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1,
        },
        textprops={
            "fontsize": 9,
        },
    )

    ax.legend(
        wedges,
        counts.index,
        title=label.replace("Participant ", "").replace(" Distribution", ""),
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )

    ax.set_title(label)
    ax.axis("equal")

    save_figure(
        fig,
        slug,
        label,
        f"Participant distribution by {label.lower()} shown as a pie chart with counts and percentages.",
    )


def plot_participant_bar_distribution(participant_df, column, label, slug):
    if participant_df.empty or column not in participant_df.columns:
        return

    counts = participant_df[column].dropna().value_counts()

    if counts.empty:
        return

    percentages = export_category_distribution(counts, slug)
    plot_df = percentages.sort_values(ascending=True)

    fig_height = max(4.2, 0.45 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    bars = ax.barh(plot_df.index, plot_df.values)

    ax.set_title(label)
    ax.set_xlabel("Participants (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)

    ax.bar_label(
        bars,
        labels=[
            f"{plot_df[label_value]:.1f}%\n(n={int(counts[label_value])})"
            for label_value in plot_df.index
        ],
        padding=3,
        fontsize=9,
    )

    save_figure(
        fig,
        slug,
        label,
        f"Participant distribution by {label.lower()} shown as percentages and counts.",
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

        rows.append(
            {
                "measure": label,
                "mean": values.mean(),
                "n": len(values),
            }
        )

    if not rows:
        return

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    plot_df = summary_df.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    bars = ax.barh(
        plot_df["measure"],
        plot_df["mean"],
    )

    ax.set_title("Participant Writing Confidence and AI Attitudes")
    ax.set_xlabel("Mean rating (1–5)")
    ax.set_ylabel("")
    ax.set_xlim(1, 5)

    ax.bar_label(
        bars,
        labels=[
            f"{row['mean']:.2f}\n(n={int(row['n'])})" for _, row in plot_df.iterrows()
        ],
        padding=3,
        fontsize=8,
    )

    save_figure(
        fig,
        slug,
        "Participant Writing Confidence and AI Attitudes",
        "Mean ratings for writing confidence and attitudes toward AI, shown with participant counts.",
    )


def plot_participant_info(participant_df):
    if participant_df.empty:
        return

    plot_participant_age_distribution(participant_df)

    plot_participant_pie_distribution(
        participant_df,
        "gender",
        "Participant Gender Distribution",
        "42_participant_gender_distribution",
    )

    plot_participant_bar_distribution(
        participant_df,
        "education",
        "Participant Education Distribution",
        "43_participant_education_distribution",
    )

    plot_participant_bar_distribution(
        participant_df,
        "nativeLanguage",
        "Participant Native Language Distribution",
        "44_participant_native_language_distribution",
    )

    plot_participant_pie_distribution(
        participant_df,
        "englishLevel",
        "Participant English Level Distribution",
        "45_participant_english_level_distribution",
    )

    plot_participant_likert_means(participant_df)
