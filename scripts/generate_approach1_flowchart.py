#!/usr/bin/env python3
"""Render the Approach 1 pipeline flowchart PNG from the Mermaid source.

The canonical diagram definition lives in docs/approach1_pipeline_flowchart.mmd.
This script draws a simple top-down flowchart in the same style as the Approach 2
Mermaid examples (rectangles + one decision diamond).

Usage (from pipeline/):
    python scripts/generate_approach1_flowchart.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
MMD_PATH = PIPELINE_ROOT / "docs" / "approach1_pipeline_flowchart.mmd"
OUT_PATH = PIPELINE_ROOT / "docs" / "approach1_pipeline_flowchart.png"

# Mirrors docs/approach1_pipeline_flowchart.mmd
FLOWCHART_TD = """
flowchart TD
    A[Load CSV cases] --> B[Select shot set: 2 high + 2 low exemplars]
    B --> C[Exclude exemplar rows from held-out test]
    C --> D{Modality tier?}
    D -->|mri_only or mri+path| E[Drop MRI-missing test cases]
    D -->|pathology_only| F[Keep all pathology cases]
    E --> G[Build few-shot prompt per test case]
    F --> G
    G --> H[SecureGPT predict JSON]
    H --> I[Validate schema + retry]
    I --> J[Save predictions JSONL/CSV]
    J --> K[Evaluate metrics vs ground truth]
""".strip()

NODES: dict[str, dict[str, object]] = {
    "A": {"label": "Load CSV cases", "kind": "box", "pos": (5.0, 10.8)},
    "B": {"label": "Select shot set:\n2 high + 2 low exemplars", "kind": "box", "pos": (5.0, 9.6)},
    "C": {"label": "Exclude exemplar rows\nfrom held-out test", "kind": "box", "pos": (5.0, 8.4)},
    "D": {"label": "Modality tier?", "kind": "diamond", "pos": (5.0, 7.0)},
    "E": {"label": "Drop MRI-missing\ntest cases", "kind": "box", "pos": (2.2, 5.5)},
    "F": {"label": "Keep all pathology\ncases", "kind": "box", "pos": (7.8, 5.5)},
    "G": {"label": "Build few-shot prompt\nper test case", "kind": "box", "pos": (5.0, 4.1)},
    "H": {"label": "SecureGPT predict JSON", "kind": "box", "pos": (5.0, 2.9)},
    "I": {"label": "Validate schema + retry", "kind": "box", "pos": (5.0, 1.7)},
    "J": {"label": "Save predictions\nJSONL/CSV", "kind": "box", "pos": (5.0, 0.5)},
    "K": {"label": "Evaluate metrics\nvs ground truth", "kind": "box", "pos": (5.0, -0.7)},
}

EDGES: list[tuple[str, str, str | None]] = [
    ("A", "B", None),
    ("B", "C", None),
    ("C", "D", None),
    ("D", "E", "mri_only or mri+path"),
    ("D", "F", "pathology_only"),
    ("E", "G", None),
    ("F", "G", None),
    ("G", "H", None),
    ("H", "I", None),
    ("I", "J", None),
    ("J", "K", None),
]


def _draw_box(ax, cx: float, cy: float, text: str, *, width: float = 2.8, height: float = 0.75) -> None:
    x, y = cx - width / 2, cy - height / 2
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#37474F",
        facecolor="#ECEFF1",
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9)


def _draw_diamond(ax, cx: float, cy: float, text: str, *, size: float = 0.95) -> None:
    pts = [
        (cx, cy + size),
        (cx + size * 1.35, cy),
        (cx, cy - size),
        (cx - size * 1.35, cy),
    ]
    patch = Polygon(pts, closed=True, linewidth=1.2, edgecolor="#37474F", facecolor="#FFF9C4")
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9)


def _node_center(node_id: str) -> tuple[float, float]:
    pos = NODES[node_id]["pos"]
    return float(pos[0]), float(pos[1])


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, label: str | None = None) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.3, shrinkA=4, shrinkB=4),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom", fontsize=7.5)


def render_flowchart(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.3, 11.5)
    ax.axis("off")
    ax.set_title("Approach 1: Few-Shot Dispersion/Relapse Pipeline", fontsize=13, fontweight="bold", pad=12)

    for node_id, meta in NODES.items():
        cx, cy = _node_center(node_id)
        label = str(meta["label"])
        if meta["kind"] == "diamond":
            _draw_diamond(ax, cx, cy, label)
        else:
            _draw_box(ax, cx, cy, label)

    for src, dst, edge_label in EDGES:
        x1, y1 = _node_center(src)
        x2, y2 = _node_center(dst)
        # Nudge arrow endpoints so lines do not overlap node bodies.
        if y1 > y2:
            y1 -= 0.38
            y2 += 0.38
        elif y2 > y1:
            y1 += 0.38
            y2 -= 0.38
        if abs(x1 - x2) > 0.5:
            if x1 < x2:
                x1 += 1.0
                x2 -= 1.0
            else:
                x1 -= 1.0
                x2 += 1.0
        _arrow(ax, x1, y1, x2, y2, edge_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not MMD_PATH.is_file():
        MMD_PATH.write_text(FLOWCHART_TD + "\n", encoding="utf-8")
    render_flowchart(OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"Mermaid source: {MMD_PATH}")


if __name__ == "__main__":
    main()
