#!/usr/bin/env python3
"""Render the Approach 2 pipeline flowchart PNG from the Mermaid source.

The canonical diagram definition lives in docs/approach2_pipeline_flowchart.mmd.
This script draws a top-down flowchart in the same style as the Approach 1 script.

Usage (from pipeline/):
    python scripts/generate_approach2_flowchart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
MMD_PATH = PIPELINE_ROOT / "docs" / "approach2_pipeline_flowchart.mmd"
OUT_PATH = PIPELINE_ROOT / "docs" / "approach2_pipeline_flowchart.png"

FLOWCHART_TD = """
flowchart TD
    A[Load CSV cases] --> B[Build outer splits]
    B --> C[Save split provenance manifest]
    C --> D[Train-only LLM extraction MRI + pathology]
    D --> E[Inner rediscovery CV on train phrases]
    E --> F[Freeze stable lexicon]
    F --> G[Rule-based recode train + test]
    G --> H[Build feature matrices]
    H --> I{MRI-derived pathway?}
    I -->|mri / combined / calibrated| J[Drop MRI-missing cases]
    I -->|pathology-only| K[Keep all pathology cases]
    J --> L[Optional pathology-calibrated MRI weights]
    K --> L
    L --> M[Optional teacher-student MRI multitask]
    M --> N[Fit scaler/model on outer-train only]
    N --> O[Predict outer-test]
    O --> P[Aggregate metrics + bootstrap CI]
    P --> Q[Generate plots + HTML reports]
""".strip()

NODES: dict[str, dict[str, object]] = {
    "A": {"label": "Load CSV cases", "kind": "box", "pos": (5.0, 15.2)},
    "B": {"label": "Build outer splits\n(repeated MC / stratified k-fold)", "kind": "box", "pos": (5.0, 14.0)},
    "C": {"label": "Save split provenance\nmanifest", "kind": "box", "pos": (5.0, 12.8)},
    "D": {"label": "Train-only LLM extraction\n(MRI + pathology)", "kind": "box", "pos": (5.0, 11.6)},
    "E": {"label": "Inner rediscovery CV\non train phrases", "kind": "box", "pos": (5.0, 10.4)},
    "F": {"label": "Freeze stable lexicon\n(+ optional feature cap)", "kind": "box", "pos": (5.0, 9.2)},
    "G": {"label": "Rule-based recode\ntrain + test", "kind": "box", "pos": (5.0, 8.0)},
    "H": {"label": "Build feature matrices", "kind": "box", "pos": (5.0, 6.8)},
    "I": {"label": "MRI-derived\npathway?", "kind": "diamond", "pos": (5.0, 5.4)},
    "J": {"label": "Drop MRI-missing\ncases", "kind": "box", "pos": (2.0, 4.0)},
    "K": {"label": "Keep all pathology\ncases", "kind": "box", "pos": (8.0, 4.0)},
    "L": {"label": "Optional pathology-calibrated\nMRI concept weights", "kind": "box", "pos": (5.0, 2.7)},
    "M": {"label": "Optional teacher-student\nMRI multitask model", "kind": "box", "pos": (5.0, 1.5)},
    "N": {"label": "Fit scaler/model on\nouter-train only", "kind": "box", "pos": (5.0, 0.3)},
    "O": {"label": "Predict outer-test", "kind": "box", "pos": (5.0, -0.9)},
    "P": {"label": "Aggregate metrics +\nbootstrap confidence intervals", "kind": "box", "pos": (5.0, -2.1)},
    "Q": {"label": "Generate plots +\nHTML review reports", "kind": "box", "pos": (5.0, -3.3)},
}

EDGES: list[tuple[str, str, str | None]] = [
    ("A", "B", None),
    ("B", "C", None),
    ("C", "D", None),
    ("D", "E", None),
    ("E", "F", None),
    ("F", "G", None),
    ("G", "H", None),
    ("H", "I", None),
    ("I", "J", "mri / combined / calibrated"),
    ("I", "K", "pathology-only"),
    ("J", "L", None),
    ("K", "L", None),
    ("L", "M", None),
    ("M", "N", None),
    ("N", "O", None),
    ("O", "P", None),
    ("P", "Q", None),
]


def _draw_box(ax, cx: float, cy: float, text: str, *, width: float = 3.0, height: float = 0.78) -> None:
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
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5)


def _draw_diamond(ax, cx: float, cy: float, text: str, *, size: float = 0.95) -> None:
    pts = [
        (cx, cy + size),
        (cx + size * 1.35, cy),
        (cx, cy - size),
        (cx - size * 1.35, cy),
    ]
    patch = Polygon(pts, closed=True, linewidth=1.2, edgecolor="#37474F", facecolor="#FFF9C4")
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5)


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
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom", fontsize=7)


def render_flowchart(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(-4.2, 16.0)
    ax.axis("off")
    ax.set_title(
        "Approach 2: Nested Lexical Discovery + Supervised ML Pipeline",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

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
        if y1 > y2:
            y1 -= 0.4
            y2 += 0.4
        elif y2 > y1:
            y1 += 0.4
            y2 -= 0.4
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
