#!/usr/bin/env python3
"""
Trim MRI + pathology report text fields in a CSV and write a new trimmed CSV.

Input:
  /Users/lukezhao/projects/onc/PROCESSED_path_cases_MRI_status.csv

Output:
  /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv

Expected columns:
  preop_MRI_text,path_report_text,dispersion_invasive_DCIS_geographic,relapse

Rules
-----
MRI text (preop_MRI_text):
  - KEEP sections: FINDINGS (any case), IMPRESSION, ENHANCEMENT PATTERN
  - REMOVE sections: INDICATION, COMPARISON, PROCEDURE, PROCEDURE COMMENTS, TECHNIQUE,
                     DATE, CONTRAST, REFERENCE
  - Do not remove anything else.

Path text (path_report_text):
  - REMOVE sections: Accession No..., OPERATION, Block Key, Reference
  - Do not remove anything else.

Notes:
  - Sections are typically separated by blank lines.
  - Section headers are typically on the first line of a section and often ALL CAPS or end with ":".
  - If a cell is empty/missing, it stays empty.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


IN_PATH = Path("/Users/lukezhao/projects/onc/PROCESSED_path_cases_MRI_status.csv")
OUT_PATH = Path("/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv")


# -----------------------------
# Section parsing / heuristics
# -----------------------------

_SECTION_SPLIT_RE = re.compile(r"\n[ \t]*\n+", flags=re.MULTILINE)


def _normalize_newlines(text: str) -> str:
    # Normalize line endings and remove obvious trailing whitespace noise
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive spaces at line ends
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def _extract_header(section: str) -> Optional[str]:
    """
    Heuristic: the header is usually the first non-empty line.
    Normalize by:
      - taking text up to ":" (if present), else full first line
      - stripping punctuation/whitespace
      - uppercasing
    """
    if not section or not section.strip():
        return None

    # Find first non-empty line
    for line in section.splitlines():
        if line.strip():
            first = line.strip()
            break
    else:
        return None

    # Common patterns: "FINDINGS:", "IMPRESSION", "Accession No: 123", "Block Key - ..."
    head = first.split(":", 1)[0].strip()
    # Also split on hyphen if it looks like a header prefix (e.g. "BLOCK KEY - ...")
    if " - " in head:
        head = head.split(" - ", 1)[0].strip()
    if "\t" in head:
        head = head.split("\t", 1)[0].strip()

    # Normalize
    head = re.sub(r"[\s\.\,;]+$", "", head)  # trailing punctuation/spaces
    head = head.upper()

    # If the first line is very long and looks like prose, treat as no header
    # (keeps section by default since rules say "Do not remove anything else")
    if len(head) > 80:
        return None

    return head or None


def _match_any_prefix(header: Optional[str], prefixes: Tuple[str, ...]) -> bool:
    if not header:
        return False
    # Compare with normalized header; allow "ACCESSION NO" to match "ACCESSION NO." etc.
    for p in prefixes:
        if header.startswith(p):
            return True
    return False


def _trim_by_sections(text: str, remove_prefixes: Tuple[str, ...]) -> str:
    """
    Split text into sections by blank lines, remove those whose *header* matches
    any prefix in remove_prefixes. Rejoin with blank lines.
    """
    text = _normalize_newlines(text).strip()
    if not text:
        return ""

    sections = [s for s in _SECTION_SPLIT_RE.split(text) if s is not None]
    kept = []

    for sec in sections:
        sec_stripped = sec.strip()
        if not sec_stripped:
            continue

        header = _extract_header(sec_stripped)
        if _match_any_prefix(header, remove_prefixes):
            continue

        kept.append(sec_stripped)

    # Re-join with exactly one blank line between sections
    out = "\n\n".join(kept).strip()
    return out


# -----------------------------
# Column-specific trimming
# -----------------------------

# MRI: remove these sections (case-insensitive handled via normalized header uppercasing)
MRI_REMOVE_PREFIXES = (
    "INDICATION",
    "COMPARISON",
    "PROCEDURE COMMENTS",
    "PROCEDURE",
    "TECHNIQUE",
    "DATE",
    "CONTRAST",
    "REFERENCE",
)

# Pathology: remove these sections
PATH_REMOVE_PREFIXES = (
    "ACCESSION NO",   # matches "Accession No", "Accession No.", "Accession No:"
    "OPERATION",
    "BLOCK KEY",
    "REFERENCE",
)


def trim_mri_text(val: Optional[str]) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    text = str(val)
    if not text.strip():
        return ""
    # Per your rules: we ONLY remove specified sections; everything else stays,
    # including FINDINGS/IMPRESSION/ENHANCEMENT PATTERN.
    return _trim_by_sections(text, MRI_REMOVE_PREFIXES)


def trim_path_text(val: Optional[str]) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    text = str(val)
    if not text.strip():
        return ""
    return _trim_by_sections(text, PATH_REMOVE_PREFIXES)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    if not IN_PATH.exists():
        print(f"ERROR: Input file not found: {IN_PATH}", file=sys.stderr)
        return 2

    df = pd.read_csv(IN_PATH, dtype=str, na_filter=False)  # keeps empty cells as ""
    required_cols = ["preop_MRI_text", "path_report_text"]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        return 2

    # Track lengths for a quick sanity check
    mri_before = df["preop_MRI_text"].fillna("").astype(str).str.len()
    path_before = df["path_report_text"].fillna("").astype(str).str.len()

    df["preop_MRI_text"] = df["preop_MRI_text"].apply(trim_mri_text)
    df["path_report_text"] = df["path_report_text"].apply(trim_path_text)

    mri_after = df["preop_MRI_text"].fillna("").astype(str).str.len()
    path_after = df["path_report_text"].fillna("").astype(str).str.len()

    # Ensure parent dir exists
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUT_PATH, index=False)

    # Simple summary
    print("Wrote trimmed CSV:")
    print(f"  {OUT_PATH}")
    print("")
    print("Text length summary (characters):")
    print(f"  MRI   total before: {int(mri_before.sum()):,}  after: {int(mri_after.sum()):,}")
    print(f"  PATH  total before: {int(path_before.sum()):,}  after: {int(path_after.sum()):,}")
    print("")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())