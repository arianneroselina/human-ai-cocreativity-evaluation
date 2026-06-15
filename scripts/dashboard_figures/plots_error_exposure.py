import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from scripts.config import (
    ERROR_ROUND_INDEX,
    EXPOSURE_LABELS,
    TABLE_DIR,
    WORKFLOW_ORDER,
)
from scripts.utils import (
    is_ai_supported_row,
    parse_bool,
    save_figure,
    workflow_label,
)


def exposure_label(value):
    fallback_labels = {
        "error_exposed": "Error exposed",
        "not_exposed": "Not exposed",
        "not_error_exposed": "Not exposed",
        "ai_supported_exposure_unknown": "AI-supported, exposure unknown",
        "unknown": "Unknown",
    }

    value_str = str(value)

    return EXPOSURE_LABELS.get(
        value_str,
        fallback_labels.get(value_str, value_str),
    )


def order_workflow_labels(labels):
    ordered_labels = [
        workflow_label(workflow)
        for workflow in WORKFLOW_ORDER
        if workflow_label(workflow) in labels
    ]

    remaining_labels = [label for label in labels if label not in ordered_labels]

    return ordered_labels + remaining_labels


def order_exposure_labels(labels):
    preferred_order = [
        "Error exposed",
        "Not exposed",
        "AI-supported, exposure unknown",
        "Unknown",
    ]

    ordered_labels = [label for label in preferred_order if label in labels]

    remaining_labels = [label for label in labels if label not in ordered_labels]

    return ordered_labels + remaining_labels


def load_interview_error_notes():
    path = "inputs/interview_error_notes.csv"

    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()


def derive_error_exposure_group(row):
    """
    Returns the participant's error exposure group.

    Priority:
    1. Use errorExposureGroup if available.
    2. Otherwise use errorExposed if available.
    3. As a fallback in the injected-error round, mark AI-supported workflows
       as exposure unknown instead of claiming actual exposure.
    """
    if "errorExposureGroup" in row.index:
        value = row.get("errorExposureGroup")

        if not pd.isna(value) and str(value).strip():
            return str(value)

    if "errorExposed" in row.index:
        return "error_exposed" if parse_bool(row.get("errorExposed")) else "not_exposed"

    if row.get("roundIndex") == ERROR_ROUND_INDEX and is_ai_supported_row(row):
        return "ai_supported_exposure_unknown"

    return "unknown"


def drop_duplicate_participant_rounds(df):
    if "participantId" not in df.columns:
        return df

    return df.drop_duplicates(
        subset=["participantId", "roundIndex"],
        keep="first",
    )


def plot_round5_workflow_exposure(df: pd.DataFrame):
    """
    Shows which workflow participants selected in round 5 and whether that
    resulted in actual AI-error exposure.
    """
    slug = "31_round5_workflow_exposure"

    required_columns = {"roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    round5_df = df[df["roundIndex"] == ERROR_ROUND_INDEX].copy()
    round5_df = round5_df.dropna(subset=["roundIndex", "workflow"])
    round5_df = drop_duplicate_participant_rounds(round5_df)

    if round5_df.empty:
        return

    round5_df["derivedErrorExposureGroup"] = round5_df.apply(
        derive_error_exposure_group,
        axis=1,
    )

    summary = (
        round5_df.groupby(["workflow", "derivedErrorExposureGroup"])
        .size()
        .reset_index(name="count")
    )

    summary["workflowLabel"] = summary["workflow"].map(workflow_label)
    summary["groupLabel"] = summary["derivedErrorExposureGroup"].map(exposure_label)

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    pivot = summary.pivot(
        index="workflowLabel", columns="groupLabel", values="count"
    ).fillna(0)

    pivot = pivot.reindex(order_workflow_labels(pivot.index))
    pivot = pivot[order_exposure_labels(pivot.columns)]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Round-5 Workflow and Actual Error Exposure")
    ax.set_xlabel("Workflow selected in round 5")
    ax.set_ylabel("Number of participants")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Exposure group")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    for container in ax.containers:
        labels = [
            f"{int(bar.get_height())}" if bar.get_height() > 0 else ""
            for bar in container
        ]

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=8,
        )

    save_figure(
        fig,
        slug,
        "Round-5 Workflow and Error Exposure",
        "Participants were exposed to the injected AI error only if actual exposure was recorded in round 5.",
    )


def plot_post_error_workflow_choices_by_exposure(df: pd.DataFrame):
    """
    Shows exact workflow choices after the possible AI-error round.
    """
    slug = "32_post_error_workflow_choices_by_exposure"

    required_columns = {"roundIndex", "workflow"}

    if not required_columns.issubset(df.columns):
        return

    choice_df = df[df["roundIndex"] > ERROR_ROUND_INDEX].copy()
    choice_df = choice_df.dropna(subset=["roundIndex", "workflow"])
    choice_df = drop_duplicate_participant_rounds(choice_df)

    if choice_df.empty:
        return

    choice_df["derivedErrorExposureGroup"] = choice_df.apply(
        derive_error_exposure_group,
        axis=1,
    )

    summary = (
        choice_df.groupby(["derivedErrorExposureGroup", "workflow"])
        .size()
        .reset_index(name="count")
    )

    if summary.empty:
        return

    summary["groupTotal"] = summary.groupby("derivedErrorExposureGroup")[
        "count"
    ].transform("sum")

    summary["percent"] = summary["count"] / summary["groupTotal"] * 100

    summary["groupLabel"] = summary["derivedErrorExposureGroup"].map(exposure_label)
    summary["workflowLabel"] = summary["workflow"].map(workflow_label)

    summary.to_csv(TABLE_DIR / f"{slug}.csv", index=False)

    pivot = summary.pivot(
        index="groupLabel", columns="workflowLabel", values="percent"
    ).fillna(0)

    pivot = pivot.reindex(order_exposure_labels(pivot.index))
    pivot = pivot[order_workflow_labels(pivot.columns)]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    pivot.plot(kind="bar", ax=ax)

    ax.set_title("Post-Error Workflow Choices by Error Exposure")
    ax.set_xlabel("Exposure group")
    ax.set_ylabel("Share of workflow choices in rounds 6–7 (%)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Chosen workflow")
    ax.set_ylim(0, 100)

    for container in ax.containers:
        labels = [
            f"{bar.get_height():.1f}%" if bar.get_height() > 0 else ""
            for bar in container
        ]

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=8,
        )

    save_figure(
        fig,
        slug,
        "Post-Error Workflow Choices by Error Exposure",
        "Distribution of workflow choices in rounds 6–7, split by whether participants were exposed to the injected AI error in round 5.",
    )


def plot_interview_coded_ai_error_summary():
    """
    Merges the interview-coded injected-error awareness summary and
    non-injected AI error type summary into one readable figure.

    Left subplot:
    - Mutually exclusive injected-error awareness categories.

    Right subplot:
    - Non-mutually-exclusive reported AI error types.
    """
    slug = "33_interview_coded_ai_error_summary"

    notes_df = load_interview_error_notes()

    if notes_df.empty:
        return

    required_columns = {
        "injectedErrorExperience",
        "reportedOtherAiErrors",
        "reportedOtherAiErrorTypes",
    }

    if not required_columns.issubset(notes_df.columns):
        return

    injected_labels = {
        "noticed": "Noticed injected AI error",
        "not_noticed": "Did not notice injected AI error",
        "not_exposed": "Not exposed to injected AI error",
    }

    awareness_summary = (
        notes_df["injectedErrorExperience"]
        .map(injected_labels)
        .value_counts()
        .reindex(
            [
                "Noticed injected AI error",
                "Did not notice injected AI error",
                "Not exposed to injected AI error",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    awareness_summary.columns = ["category", "participantCount"]
    awareness_summary["percentage"] = (
        awareness_summary["participantCount"] / len(notes_df) * 100
    ).round(2)

    other_ai_error_count = int(
        notes_df["reportedOtherAiErrors"].apply(parse_bool).sum()
    )

    awareness_summary.loc[len(awareness_summary)] = {
        "category": "Reported other AI errors",
        "participantCount": other_ai_error_count,
        "percentage": round(other_ai_error_count / len(notes_df) * 100, 2),
    }

    error_type_rows = []

    for _, row in notes_df.iterrows():
        raw_types = row.get("reportedOtherAiErrorTypes")

        if pd.isna(raw_types) or not str(raw_types).strip():
            continue

        for error_type in str(raw_types).split(";"):
            cleaned_error_type = error_type.strip()

            if not cleaned_error_type:
                continue

            error_type_rows.append(
                {
                    "errorType": cleaned_error_type,
                    "participantId": row.get("participantId"),
                }
            )

    labels = {
        "counting_constraints": "Counting constraints",
        "requirement_following": "Requirement following",
        "formatting": "Formatting",
        "time_conversion": "Time conversion",
        "required_words": "Required words",
        "case_sensitivity": "Case sensitivity",
        "quality_degradation": "Quality degradation",
    }

    if error_type_rows:
        error_type_df = pd.DataFrame(error_type_rows)

        error_type_df["errorTypeLabel"] = (
            error_type_df["errorType"].map(labels).fillna(error_type_df["errorType"])
        )

        error_type_summary = (
            error_type_df.groupby("errorTypeLabel")["participantId"]
            .nunique()
            .reset_index(name="participantCount")
            .sort_values("participantCount", ascending=True)
        )

        error_type_summary["percentage"] = (
            error_type_summary["participantCount"] / len(notes_df) * 100
        ).round(2)
    else:
        error_type_summary = pd.DataFrame(
            columns=["errorTypeLabel", "participantCount", "percentage"]
        )

    awareness_summary.to_csv(
        TABLE_DIR / f"{slug}_awareness.csv",
        index=False,
    )

    error_type_summary.to_csv(
        TABLE_DIR / f"{slug}_error_types.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 5.2),
    )

    awareness_plot_df = awareness_summary.iloc[::-1]

    awareness_bars = axes[0].barh(
        awareness_plot_df["category"],
        awareness_plot_df["participantCount"],
    )

    axes[0].set_title("Injected AI-error awareness")
    axes[0].set_xlabel("Number of participants")
    axes[0].set_ylabel("")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].bar_label(
        awareness_bars,
        labels=[
            f"{int(row['participantCount'])} ({row['percentage']:.1f}%)"
            for _, row in awareness_plot_df.iterrows()
        ],
        padding=3,
        fontsize=8,
    )

    if not error_type_summary.empty:
        error_type_bars = axes[1].barh(
            error_type_summary["errorTypeLabel"],
            error_type_summary["participantCount"],
        )

        axes[1].bar_label(
            error_type_bars,
            labels=[
                f"{int(row['participantCount'])} ({row['percentage']:.1f}%)"
                for _, row in error_type_summary.iterrows()
            ],
            padding=3,
            fontsize=8,
        )

    axes[1].set_title("Reported non-injected AI error types")
    axes[1].set_xlabel("Number of participants")
    axes[1].set_ylabel("")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(
        "Interview-coded AI Error Summary",
        fontsize=14,
    )

    fig.text(
        0.5,
        0.01,
        "Note: Awareness categories describe injected-error experience. Reported non-injected AI error types are not mutually exclusive.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    save_figure(
        fig,
        slug,
        "Interview-coded AI Error Summary",
        "Participant-level interview coding of injected AI-error awareness and reported non-injected AI error types.",
    )


def plot_error_exposure(df: pd.DataFrame):
    plot_round5_workflow_exposure(df)
    plot_post_error_workflow_choices_by_exposure(df)
    plot_interview_coded_ai_error_summary()
