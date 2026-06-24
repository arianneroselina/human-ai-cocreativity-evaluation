from scripts.config import (
    MAIN_ROUND_INDICES,
    WORKFLOW_LABELS,
    ROUND_LABELS,
    EXPOSURE_LABELS, EVALUATOR_LABELS,
)
from scripts.utils import parse_bool


def get_main_rounds(df):
    return df[df["roundIndex"].isin(MAIN_ROUND_INDICES)].copy()


def get_main_round_position(round_index):
    mapping = {
        5: 1,
        6: 2,
        7: 3,
    }
    return mapping.get(round_index)


def get_complete_main_round_participants(df):
    participant_rounds = df.groupby("participantId")["roundIndex"].apply(set)

    complete_ids = participant_rounds[
        participant_rounds.apply(
            lambda rounds: set(MAIN_ROUND_INDICES).issubset(rounds)
        )
    ].index

    return df[df["participantId"].isin(complete_ids)].copy()


def workflow_display_name(workflow):
    return WORKFLOW_LABELS.get(workflow, workflow)


def round_display_name(round_index):
    return ROUND_LABELS.get(round_index, f"Round {round_index}")


def exposure_display_name(group: str) -> str:
    return EXPOSURE_LABELS.get(group, str(group).replace("_", " ").title())


def evaluator_display_name(evaluator_id: str) -> str:
    return EVALUATOR_LABELS.get(str(evaluator_id), str(evaluator_id))


def shade_main_rounds(
    ax,
    start_round=5,
    end_round=7,
    label="Main rounds (5–7)",
    color="#f2f2f2",
    alpha=0.8,
    label_y=0.03,
):
    """
    Adds a light background shade for the main rounds.

    Use this for line plots where the x-axis is the actual round number.
    Example: rounds 1, 2, 3, 4, 5, 6, 7.
    """
    left = start_round - 0.5
    right = end_round + 0.5
    center = (start_round + end_round) / 2

    ax.axvspan(
        left,
        right,
        color=color,
        alpha=alpha,
        zorder=0,
    )

    if label:
        ax.text(
            center,
            label_y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.75,
            },
        )


def shade_main_rounds_for_bar_axis(
    ax,
    round_indices,
    start_round=5,
    end_round=7,
    label="Main rounds (5–7)",
    color="#f2f2f2",
    alpha=0.8,
    label_y=0.03,
):
    """
    Adds background shading for bar plots where rounds are shown as categorical bars.

    Pandas bar plots use positions 0, 1, 2, ... instead of the actual round numbers.
    """
    round_positions = {
        int(round_index): position for position, round_index in enumerate(round_indices)
    }

    positions = [
        position
        for round_index, position in round_positions.items()
        if start_round <= round_index <= end_round
    ]

    if not positions:
        return

    left = min(positions) - 0.5
    right = max(positions) + 0.5
    center = (left + right) / 2

    ax.axvspan(
        left,
        right,
        color=color,
        alpha=alpha,
        zorder=0,
    )

    if label:
        ax.text(
            center,
            label_y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.75,
            },
        )


def annotate_injected_error_round(ax, error_idx, y_top=5.0, text_y=5.25):
    ax.axvline(
        error_idx,
        linestyle="--",
        linewidth=1,
        zorder=3,
    )

    ax.annotate(
        "Injected AI error",
        xy=(error_idx, y_top),
        xytext=(error_idx + 0.12, text_y),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=8,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.8,
        },
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.9,
        },
    )


def is_ai_supported_row(row):
    if "isAiSupportedWorkflow" in row.index:
        return parse_bool(row["isAiSupportedWorkflow"])

    return row.get("workflow") in ["ai", "human_ai", "ai_human"]
