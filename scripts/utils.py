import json
import matplotlib.pyplot as plt
import pandas as pd

from scripts.config import WORKFLOW_LABELS, FIGURE_DIR

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = []


def workflow_label(value):
    return WORKFLOW_LABELS.get(str(value), str(value))


def save_figure(fig, slug, title, description):
    png_path = FIGURE_DIR / f"{slug}.png"
    pdf_path = FIGURE_DIR / f"{slug}.pdf"
    svg_path = FIGURE_DIR / f"{slug}.svg"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    MANIFEST.append(
        {
            "slug": slug,
            "title": title,
            "description": description,
            "pngUrl": f"/research-dashboard/figures/{slug}.png",
            "pdfUrl": f"/research-dashboard/figures/{slug}.pdf",
            "svgUrl": f"/research-dashboard/figures/{slug}.svg",
        }
    )


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


def save_manifest():
    manifest_path = FIGURE_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(MANIFEST, file, indent=2)


def ensure_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def parse_bool(value):
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    return normalized in {"true", "t", "1", "yes", "y"}


def parse_bool_or_none(value):
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "t", "1", "yes", "y"}:
        return True

    if normalized in {"false", "f", "0", "no", "n"}:
        return False

    return None


def is_ai_supported_row(row):
    if "isAiSupportedWorkflow" in row.index:
        return parse_bool(row["isAiSupportedWorkflow"])

    return row.get("workflow") in ["ai", "human_ai", "ai_human"]
