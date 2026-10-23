#!/usr/bin/env python3
"""Copy selected columns from one CSV into a new CSV.

Cell text and row order are left unchanged. Only the columns listed in
COLUMNS are written, in that order.

Usage:
    python filter_data.py INPUT.csv OUTPUT.csv
"""

import csv
import sys
from pathlib import Path

COLUMNS = [
    "case_id",
    "diagnostic_MRI_text",
    "preop_MRI_text",
    "path_report_text",
    "relapse",
    "RFS",
    "dmax_invasive",
    "dispersion_invasive_DCIS_geographic",
    "index_side",
    "RCB_Continuous",
    "RCB_Class",
]


def main(argv):
    if len(argv) != 3:
        print("usage: filter_data.py INPUT.csv OUTPUT.csv", file=sys.stderr)
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2])

    try:
        with in_path.open(newline="", encoding="utf-8-sig") as src:
            reader = csv.reader(src)
            try:
                header = next(reader)
            except StopIteration:
                print(f"Error: {in_path} is empty", file=sys.stderr)
                return 1

            index = {}
            for i, name in enumerate(header):
                if name in index:
                    print(
                        f"Error: duplicate column {name!r} in {in_path}",
                        file=sys.stderr,
                    )
                    return 1
                index[name] = i

            missing = [name for name in COLUMNS if name not in index]
            if missing:
                print(
                    f"Error: {in_path} is missing columns: {', '.join(missing)}",
                    file=sys.stderr,
                )
                return 1

            positions = [index[name] for name in COLUMNS]

            with out_path.open("w", newline="", encoding="utf-8") as dst:
                writer = csv.writer(dst, lineterminator="\n")
                writer.writerow(COLUMNS)
                for row in reader:
                    writer.writerow(
                        [row[i] if i < len(row) else "" for i in positions]
                    )
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except UnicodeError as exc:
        print(f"Error decoding {in_path} as UTF-8: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
