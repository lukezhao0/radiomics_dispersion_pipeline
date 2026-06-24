"""Standalone CLI for Approach 2 HTML report generation."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import webbrowser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Approach 2 results, interpretability, and missed-case HTML review reports.",
    )
    parser.add_argument(
        "--run-dir",
        "--out_dir",
        dest="run_dir",
        required=True,
        help="Existing nested-evaluation output directory with nested_outer_* artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for HTML outputs (default: same as --run-dir).",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional source CSV for MRI/pathology availability flags in missed-case analysis.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate reports even if HTML files already exist.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the results HTML report in a browser after generation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from approach2.reports import generate_all_reports

    args = build_parser().parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir or run_dir)
    if output_dir != run_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name in os.listdir(run_dir):
            src = os.path.join(run_dir, name)
            dst = os.path.join(output_dir, name)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
            elif os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    paths = generate_all_reports(output_dir, csv_path=args.csv_path, force=args.force)
    print("[DONE] Report generation complete.")
    for key, path in paths.items():
        if path:
            print(f"  {key}: {path}")

    if args.open and paths.get("results_html") and os.path.isfile(paths["results_html"]):
        webbrowser.open(f"file://{paths['results_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
