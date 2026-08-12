"""Participant demographics, writing confidence, and AI attitudes (figures 51-57).

51  Age distribution (histogram)
52  Gender distribution (pie)
53  Education distribution (bar)
54  Native language distribution (bar)
55  English level distribution (pie)
56  Writing confidence mean (bar)
57  AI attitude Likert means (bar)
"""

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
    slug = "51_participant_age_distribution"

    if "age" not in participant_df.columns:
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
        linewidth=1,
        label=f"Mean = {mean_age:.1f}",
    )

    ax.axvline(
        median_age,
        color="darkgreen",
        linestyle=":",
        linewidth=1,
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


def normalize_native_language(series: pd.Series) -> pd.Series:
    """Normalize native-language labels to sentence capitalization."""
    normalized = series.astype("string").str.strip().str.lower().str.capitalize()

    return normalized.mask(normalized.eq("")).dropna()


def plot_participant_native_language_distribution(participant_df):
    slug = "54_participant_native_language_distribution"

    if "nativeLanguage" not in participant_df.columns:
        return

    languages = normalize_native_language(participant_df["nativeLanguage"])

    if languages.empty:
        return

    # Count after normalization so, for example, KANNADA and Kannada
    # are treated as the same language.
    normalized_counts = languages.value_counts()

    # Group only languages represented by one participant.
    singleton_languages = normalized_counts[normalized_counts == 1].index.tolist()

    display_languages = languages.mask(
        languages.isin(singleton_languages),
        "Other languages",
    )

    counts = display_languages.value_counts()
    percentages = export_category_distribution(counts, slug)

    # Export the normalization and grouping details for transparency.
    grouping_df = pd.DataFrame(
        {
            "language": normalized_counts.index,
            "count": normalized_counts.values,
            "display_category": [
                ("Other languages" if language in singleton_languages else language)
                for language in normalized_counts.index
            ],
        }
    )

    grouping_df.to_csv(
        TABLE_DIR / f"{slug}_grouping.csv",
        index=False,
    )

    plot_df = pd.DataFrame(
        {
            "percentage": percentages,
            "count": counts,
        }
    ).sort_values(
        ["percentage", "count"],
        ascending=True,
    )

    fig_height = max(4.0, 0.52 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(7.6, fig_height))

    bars = ax.barh(
        plot_df.index,
        plot_df["percentage"],
    )

    ax.set_title("Participant Native Languages")
    ax.set_xlabel("Participants (%)")
    ax.set_ylabel("")

    # Avoid the large unused area produced by a fixed 0–100 axis.
    max_percentage = plot_df["percentage"].max()
    ax.set_xlim(0, min(100, max_percentage + 12))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.bar_label(
        bars,
        labels=[
            f"{row['percentage']:.1f}% (n={int(row['count'])})"
            for _, row in plot_df.iterrows()
        ],
        padding=4,
        fontsize=9,
    )

    other_description = (
        ", ".join(sorted(singleton_languages)) if singleton_languages else "none"
    )

    save_figure(
        fig,
        slug,
        "Participant Native Languages",
        "Distribution of participants' normalized native-language responses. "
        f"Languages represented by one participant were grouped as "
        f"'Other languages': {other_description}.",
    )


def plot_participant_pie_distribution(participant_df, column, label, slug):
    if column not in participant_df.columns:
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
        f"Participant distribution by {label.lower()} shown as a pie chart "
        "with counts and percentages.",
    )


def plot_participant_bar_distribution(participant_df, column, label, slug):
    if column not in participant_df.columns:
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


def plot_participant_writing_confidence(participant_df):
    """Plot the distribution of writing-confidence ratings."""
    slug = "56_participant_writing_confidence"

    confidence_items = [
        (column, label)
        for column, label in PARTICIPANT_LIKERT_COLUMNS.items()
        if "confidence" in label.lower()
    ]

    if not confidence_items:
        return

    column, _ = confidence_items[0]

    if column not in participant_df.columns:
        return

    values = pd.to_numeric(
        participant_df[column],
        errors="coerce",
    ).dropna()

    # Retain only valid five-point scale responses.
    values = values[values.between(1, 5)].round().astype(int)

    if values.empty:
        return

    counts = values.value_counts().reindex(range(1, 6), fill_value=0).sort_index()

    percentages = counts / counts.sum() * 100

    distribution_df = pd.DataFrame(
        {
            "rating": counts.index,
            "count": counts.values,
            "percentage": percentages.round(2).values,
        }
    )

    distribution_df.to_csv(
        TABLE_DIR / f"{slug}.csv",
        index=False,
    )

    summary_df = pd.DataFrame(
        {
            "n": [len(values)],
            "mean": [values.mean()],
            "standard_deviation": [values.std()],
            "median": [values.median()],
        }
    )

    summary_df.to_csv(
        TABLE_DIR / f"{slug}_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    bars = ax.bar(
        counts.index,
        percentages.values,
    )

    ax.bar_label(
        bars,
        labels=[
            f"{percentage:.1f}%\n(n={count})" if count > 0 else ""
            for count, percentage in zip(
                counts.values,
                percentages.values,
            )
        ],
        padding=3,
        fontsize=9,
    )

    mean_rating = values.mean()

    ax.axvline(
        mean_rating,
        linestyle="--",
        linewidth=1,
        label=f"Mean = {mean_rating:.2f}",
    )

    ax.set_title("Confidence in Writing Under Time Pressure")
    ax.set_xlabel("Confidence rating (1 = low, 5 = high)")
    ax.set_ylabel("Participants (%)")
    ax.set_xticks(range(1, 6))

    upper_limit = max(percentages.max() + 10, 40)
    ax.set_ylim(0, min(100, upper_limit))

    ax.legend(
        title=f"n = {len(values)}",
        loc="upper left",
    )

    save_figure(
        fig,
        slug,
        "Confidence in Writing Under Time Pressure",
        "Distribution of participants' pre-study confidence ratings for "
        "writing under time pressure. The dashed line indicates the mean.",
    )


def plot_likert_mean_chart(
    summary_df: pd.DataFrame,
    slug: str,
    title: str,
    description: str,
    figsize: tuple[float, float],
) -> None:
    """Plot and export mean ratings for a group of Likert-scale items."""
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df.to_csv(
        TABLE_DIR / f"{slug}.csv",
        index=False,
    )

    plot_df = summary_df.sort_values(
        "mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        plot_df["measure"],
        plot_df["mean"],
    )

    ax.set_title(title)
    ax.set_xlabel("Mean rating (1–5)")
    ax.set_ylabel("")
    ax.set_xlim(1, 5)

    ax.bar_label(
        bars,
        labels=[
            f"{row['mean']:.2f} (n={int(row['n'])})" for _, row in plot_df.iterrows()
        ],
        padding=4,
        fontsize=9,
    )

    save_figure(
        fig,
        slug,
        title,
        description,
    )


def plot_participant_ai_attitude_means(participant_df):
    slug = "57_participant_ai_attitude_means"

    rows = []

    for column, label in PARTICIPANT_LIKERT_COLUMNS.items():
        # Writing confidence is shown separately.
        if "confidence" in label.lower():
            continue

        if column not in participant_df.columns:
            continue

        values = pd.to_numeric(
            participant_df[column],
            errors="coerce",
        ).dropna()

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

    plot_likert_mean_chart(
        summary_df=summary_df,
        slug=slug,
        title="Participant Attitudes Toward AI",
        description=(
            "Mean ratings for the four participant AI-attitude items, "
            "shown with the number of responses."
        ),
        figsize=(8.4, 4.2),
    )


def plot_participant_info(participant_df):
    if participant_df.empty:
        return

    plot_participant_age_distribution(participant_df)

    plot_participant_pie_distribution(
        participant_df,
        "gender",
        "Participant Gender Distribution",
        "52_participant_gender_distribution",
    )

    plot_participant_bar_distribution(
        participant_df,
        "education",
        "Participant Education Distribution",
        "53_participant_education_distribution",
    )

    plot_participant_native_language_distribution(participant_df)

    plot_participant_pie_distribution(
        participant_df,
        "englishLevel",
        "Participant English Level Distribution",
        "55_participant_english_level_distribution",
    )

    plot_participant_writing_confidence(participant_df)
    plot_participant_ai_attitude_means(participant_df)
