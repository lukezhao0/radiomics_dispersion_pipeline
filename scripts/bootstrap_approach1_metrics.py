#!/usr/bin/env python3
"""Compute Approach 1 bootstrap metric CIs from a saved predictions CSV (no API calls)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from approach1.config import DEFAULT_BOOTSTRAP_N, DEFAULT_BOOTSTRAP_SEED
from approach1.evaluation.bootstrap import BOOTSTRAP_CSV_FILENAME, compute_and_save_bootstrap_cis


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute case-level bootstrap 95%% CIs for Approach 1 metrics from "
            "predictions_testing_cases.csv (no LLM/API calls)."
        )
    )
    parser.add_argument(
        "predictions_csv",
        help="Path to predictions_testing_cases.csv",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default=None,
        help="Directory for bootstrap_metric_cis.csv/json (default: same directory as predictions CSV).",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=DEFAULT_BOOTSTRAP_N,
        help=f"Number of bootstrap replicates (default: {DEFAULT_BOOTSTRAP_N}; use 0 to disable).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Random seed (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    args = parser.parse_args()

    predictions_csv = os.path.abspath(args.predictions_csv)
    if not os.path.isfile(predictions_csv):
        print(f"[ERROR] Predictions CSV not found: {predictions_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(args.outdir or os.path.dirname(predictions_csv))
    pred_df = pd.read_csv(predictions_csv)

    if args.bootstrap_n <= 0:
        print("[BOOTSTRAP] Disabled (--bootstrap-n <= 0); nothing to write.")
        return

    summary = compute_and_save_bootstrap_cis(
        pred_df,
        out_dir,
        n_bootstrap=args.bootstrap_n,
        random_seed=args.bootstrap_seed,
    )
    csv_path = summary["paths"]["csv"]
    print(f"[BOOTSTRAP] Wrote {summary['n_metrics']} metrics to {csv_path}")
    print(f"[BOOTSTRAP] Also wrote {os.path.join(out_dir, BOOTSTRAP_CSV_FILENAME.replace('.csv', '.json'))}")


if __name__ == "__main__":
    main()
