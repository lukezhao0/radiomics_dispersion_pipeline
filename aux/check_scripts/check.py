#!/usr/bin/env python3
"""Check that two CSV files hold identical data, row for row, in the same order.

Only the columns listed in COLUMNS are loaded and compared. Each file's text
encoding is detected independently, so the two files do not have to agree.
"""

import argparse
import io
import re
import sys
import unicodedata
from collections import Counter

import pandas as pd

DEFAULT_CSV_1 = "/Users/lukezhao/projects/onc/data/UPDATED_path_MRI_cases.csv"
DEFAULT_CSV_2 = "/Users/lukezhao/projects/onc/data/path_cases_MRI_status.csv"

COLUMNS = [
    "case_id",
    "diagnostic_MRI_exists",
    "preop_MRI_exists",
    "relapse",
    "dispersion_invasive_DCIS_geographic",
]

NON_ASCII = r"[^\x00-\x7F]"

# Considered in order once UTF-8 has been ruled out. Each of these maps all 256
# byte values, so none of them can fail -- they have to be told apart by what
# they produce, not by whether they decode. mac_roman leads because these
# exports come from Excel on macOS, where byte 0xCA is a non-breaking space.
SINGLE_BYTE_CANDIDATES = ("mac_roman", "cp1252", "latin-1")

BOMS = (
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16"),
    (b"\xff\xfe", "utf-16"),
)


def describe_decode_error(data, exc):
    """Locate a UnicodeDecodeError in the file: line number and nearby bytes."""
    i = exc.start
    line_no = data.count(b"\n", 0, i) + 1
    line_start = data.rfind(b"\n", 0, i) + 1
    line_end = data.find(b"\n", i)
    if line_end == -1:
        line_end = len(data)
    snippet = data[max(line_start, i - 40) : min(line_end, i + 40)]
    return (
        f"byte 0x{data[i]:02x} at offset {i} (csv line {line_no}) is not valid {exc.encoding}\n"
        f"    context: {snippet!r}"
    )


def profile_non_ascii(text):
    """Count non-ASCII characters in text, and how many of them are letters."""
    found = [c for c in text if ord(c) > 127]
    letters = sum(1 for c in found if unicodedata.category(c).startswith("L"))
    return len(found), letters


def detect_encoding(data):
    """Return (encoding, how_it_was_chosen, candidates) for raw CSV bytes."""
    for bom, encoding in BOMS:
        if data.startswith(bom):
            return encoding, f"detected: {encoding} byte order mark", []

    try:
        data.decode("utf-8")
        return "utf-8", "detected: decodes as valid UTF-8", []
    except UnicodeDecodeError:
        pass

    candidates = []
    for encoding in SINGLE_BYTE_CANDIDATES:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        total, letters = profile_non_ascii(text)
        candidates.append((encoding, total, letters))

    # Statistical detectors routinely confuse the single-byte codecs, so decide
    # on what the decoding actually yields. In English-language clinical
    # exports a stray non-ASCII *letter* almost always means the codec is
    # wrong, while typographic characters (non-breaking space, curly quotes,
    # dashes) mean it is right.
    typographic = [c for c in candidates if c[2] == 0]
    if typographic:
        encoding = typographic[0][0]
        return encoding, "detected: yields no stray non-ASCII letters", candidates

    # Every candidate produced letters, so the text probably contains genuine
    # accented characters. Let a statistical detector break the tie, confined
    # to the candidates above so it cannot wander off to an exotic codec.
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        pass
    else:
        best = from_bytes(data, cp_isolation=list(SINGLE_BYTE_CANDIDATES)).best()
        if best is not None and best.encoding:
            return best.encoding, "detected: charset_normalizer tie-break", candidates

    return candidates[0][0], "fallback: first candidate, unconfirmed", candidates


def read_text(path, encoding):
    """Read path as text, detecting its encoding unless one was forced."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise SystemExit(f"Error opening {path}: {exc}")

    if encoding == "auto":
        encoding, how, candidates = detect_encoding(data)
    else:
        how, candidates = "forced via --encoding", []

    try:
        return data.decode(encoding), encoding, how, candidates
    except LookupError:
        raise SystemExit(f"Unknown encoding {encoding!r} requested for {path}")
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"Error decoding {path} as {encoding} ({how}):\n"
            f"    {describe_decode_error(data, exc)}\n"
            f"    Re-run with --encoding to force a different codec."
        )


def load(path, encoding):
    """Load only COLUMNS from path as raw strings, in COLUMNS order."""
    text, encoding, how, candidates = read_text(path, encoding)
    # dtype=str + keep_default_na=False avoids pandas inferring different types
    # for the same value in the two files (e.g. 1 vs 1.0, empty vs NaN).
    try:
        df = pd.read_csv(
            io.StringIO(text), usecols=COLUMNS, dtype=str, keep_default_na=False
        )
    except ValueError as exc:
        raise SystemExit(f"Error reading {path}: {exc}")
    return df[COLUMNS], encoding, how, candidates


def normalize(df):
    """Fold away encoding and whitespace artifacts so values compare on content.

    NFKC maps compatibility characters onto their plain equivalents, most
    importantly U+00A0 (byte 0xCA in Mac Roman) onto a regular space. Splitting
    and rejoining then trims the ends and collapses internal whitespace runs.
    """
    return df.map(lambda s: " ".join(unicodedata.normalize("NFKC", s).split()))


def non_ascii_summary(df):
    """Return (cells per column, counts per character) for non-ASCII content."""
    per_column = {}
    characters = Counter()
    for column in COLUMNS:
        matches = df[column].str.contains(NON_ASCII, regex=True)
        if matches.any():
            per_column[column] = int(matches.sum())
            for value in df[column][matches]:
                characters.update(re.findall(NON_ASCII, value))
    return per_column, characters


def describe_file(label, path, df, encoding, how, candidates):
    print(f"{label}: {path}")
    print(f"        {len(df)} rows, encoding {encoding} ({how})")
    # Show the alternatives whenever the choice was a judgement call, so a
    # wrong guess is visible rather than silently shaping the comparison.
    for name, total, letters in candidates:
        marker = "->" if name == encoding else "  "
        print(f"        {marker} {name}: {total} non-ASCII char(s), {letters} letter(s)")


def report_non_ascii(label, df):
    per_column, characters = non_ascii_summary(df)
    if not per_column:
        return
    print(f"{label}: non-ASCII characters present in the compared columns")
    for column, count in per_column.items():
        print(f"    {column}: {count} cell(s)")
    for character, count in characters.most_common():
        try:
            name = unicodedata.name(character)
        except ValueError:
            name = "unnamed"
        print(f"    U+{ord(character):04X} {name} x{count}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv1", default=DEFAULT_CSV_1, help="First CSV file")
    parser.add_argument("--csv2", default=DEFAULT_CSV_2, help="Second CSV file")
    parser.add_argument(
        "--encoding",
        default="auto",
        help="Force a codec for both files (e.g. utf-8, mac_roman, cp1252). "
        "Default 'auto' detects each file separately.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Compare decoded text verbatim instead of applying NFKC and "
        "whitespace normalization",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=50,
        help="Maximum number of mismatched rows to print (default: 50)",
    )
    args = parser.parse_args()

    raw1, encoding1, how1, candidates1 = load(args.csv1, args.encoding)
    raw2, encoding2, how2, candidates2 = load(args.csv2, args.encoding)

    describe_file("file 1", args.csv1, raw1, encoding1, how1, candidates1)
    describe_file("file 2", args.csv2, raw2, encoding2, how2, candidates2)
    print(f"columns compared: {', '.join(COLUMNS)}")
    print()

    report_non_ascii("file 1", raw1)
    report_non_ascii("file 2", raw2)

    row_count_matches = len(raw1) == len(raw2)
    n = min(len(raw1), len(raw2))
    if not row_count_matches:
        print(f"ROW COUNT MISMATCH: {len(raw1)} vs {len(raw2)}")
        print(f"Comparing the first {n} rows of each file.")
        print()

    raw_a = raw1.head(n).reset_index(drop=True)
    raw_b = raw2.head(n).reset_index(drop=True)
    a = raw_a if args.no_normalize else normalize(raw_a)
    b = raw_b if args.no_normalize else normalize(raw_b)

    differs = a.ne(b)
    mismatched_rows = differs.any(axis=1)
    total_mismatched_rows = int(mismatched_rows.sum())

    if not args.no_normalize:
        hidden = int((raw_a.ne(raw_b) & ~differs).to_numpy().sum())
        if hidden:
            print(
                f"note: {hidden} cell(s) differ byte-for-byte but match after "
                f"normalization; counted as equal. Use --no-normalize to see them."
            )
            print()

    if total_mismatched_rows:
        print(f"{total_mismatched_rows} of {n} compared rows differ.")
        print()
        print("mismatches per column:")
        for column in COLUMNS:
            print(f"  {column}: {int(differs[column].sum())}")
        print()

        shown = 0
        for i in differs.index[mismatched_rows]:
            if shown >= args.max_report:
                print(f"... {total_mismatched_rows - shown} more mismatched row(s) not shown.")
                break
            # +2 maps a 0-based data row to its line number in the CSV (header is line 1).
            print(f"row {i} (csv line {i + 2}):")
            for column in COLUMNS:
                if differs.at[i, column]:
                    print(f"    {column}: {a.at[i, column]!r} != {b.at[i, column]!r}")
            shown += 1
        print()

    if row_count_matches and total_mismatched_rows == 0:
        print("PASS: both files are identical across the compared columns, in the same order.")
        return 0

    print("FAIL: the files are not identical.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
