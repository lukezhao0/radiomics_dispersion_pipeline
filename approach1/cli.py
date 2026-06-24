"""CLI entry point for the approach1 pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from . import config
from .api import confirm_before_full_run, configure_api, preflight_check, print_cumulative_report, reset_cost_tracker
from .data import load_cases
from .logging_setup import Tee
from .evaluation.results_report import build_approach1_results_html
from .orchestration import run_one_config, save_aggregate_summary
from .splits import build_run_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureGPT 3-tier dispersion/relapse pipeline + evaluation")
    parser.add_argument("--csv-path", "-c", default=config.CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--outdir", "-o", default=config.OUT_DIR, help="Root directory to write all shotset/tier outputs.")
    parser.add_argument("--env-path", default=config.ENV_PATH, help="Path to .env containing SANDBOX_API_KEY.")
    parser.add_argument("--deployment", default=config.DEPLOYMENT, help="SecureGPT/Azure deployment name.")
    parser.add_argument("--api-version", default=config.API_VERSION, help="Azure OpenAI API version.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help="Sampling temperature for chat completions (default: 0 for deterministic JSON).",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip the interactive a-priori cost confirmation prompt.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the initial small API connectivity test.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-case JSONL checkpoints and completed config markers in --outdir (default: enabled).",
    )
    parser.add_argument(
        "--skip-completed-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, skip shotset/modality folders with a valid completed checkpoint (default: enabled).",
    )
    parser.add_argument(
        "--force-rerun-cases",
        action="store_true",
        help="Ignore per-case checkpoints and call SecureGPT again for every test case in non-skipped configs.",
    )
    parser.add_argument(
        "--results-report-only",
        action="store_true",
        help="Build approach1_results_report.html from existing artifacts in --outdir and exit (no API calls).",
    )
    args = parser.parse_args()

    if args.results_report_only:
        build_approach1_results_html(args.outdir)
        return

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "run.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_mode = "a" if args.resume and os.path.isfile(log_path) else "w"

    with open(log_path, log_mode, encoding="utf-8") as log_f:
        sys.stdout = Tee(original_stdout, log_f)
        sys.stderr = Tee(original_stderr, log_f)
        try:
            if log_mode == "a":
                print("\n" + "=" * 80)
                print(f"[RESUME SESSION] {datetime.now().isoformat(timespec='seconds')}")
            print("=" * 80)
            print("[START] SecureGPT 3-tier dispersion/relapse pipeline + evaluation")
            print(f"[START] CSV_PATH={args.csv_path}")
            print(f"[START] OUT_DIR={args.outdir}")
            print(f"[START] ENV_PATH={args.env_path}")
            print(f"[START] DEPLOYMENT={args.deployment}")
            print(f"[START] API_VERSION={args.api_version}")
            print(f"[START] TEMPERATURE={args.temperature}")
            print(f"[START] LOG_PATH={log_path}")
            print(f"[START] RESUME={args.resume}")
            print(f"[START] SKIP_COMPLETED_CONFIGS={args.skip_completed_configs}")
            print(f"[START] FORCE_RERUN_CASES={args.force_rerun_cases}")
            print("=" * 80)

            config.TEMPERATURE = float(args.temperature)
            configure_api(args.env_path, args.deployment, args.api_version)
            df = load_cases(args.csv_path)
            run_configs = build_run_configs(df, args.outdir)

            confirm_before_full_run(
                run_configs,
                assume_yes=args.yes,
                resume=args.resume,
                skip_completed_configs=args.skip_completed_configs,
                force_rerun_cases=args.force_rerun_cases,
            )
            if not args.skip_preflight:
                reset_cost_tracker()
                preflight_check()
                print_cumulative_report()

            aggregate_summaries: List[Dict[str, Any]] = []
            for rc in run_configs:
                _, metrics = run_one_config(
                    rc,
                    resume=args.resume,
                    skip_completed_configs=args.skip_completed_configs,
                    force_rerun_cases=args.force_rerun_cases,
                )
                aggregate_summaries.append({
                    "shotset_name": rc.shotset_name,
                    "modality": rc.modality,
                    "n_skipped_missing_mri": len(rc.skipped_missing_mri),
                    "metrics": metrics,
                })

            save_aggregate_summary(args.outdir, aggregate_summaries)
            build_approach1_results_html(args.outdir)
            print("[END] All shotset/tier runs complete.")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
