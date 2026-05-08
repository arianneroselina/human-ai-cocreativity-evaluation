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

    MANIFEST.append({
        "slug": slug,
        "title": title,
        "description": description,
        "pngUrl": f"/research-dashboard/figures/{slug}.png",
        "pdfUrl": f"/research-dashboard/figures/{slug}.pdf",
        "svgUrl": f"/research-dashboard/figures/{slug}.svg",
    })


def save_manifest():
    manifest_path = FIGURE_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(MANIFEST, file, indent=2)


def ensure_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

