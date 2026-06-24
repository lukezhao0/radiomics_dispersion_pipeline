"""CLI entry point for the approach1 pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from . import config
from .api import (
    aggregate_pipeline_cost_report,
    confirm_before_full_run,
    configure_api,
    preflight_check,
    print_cumulative_report,
    print_cumulative_report_snapshot,
    reset_cost_tracker,
    save_pipeline_cost_report,
)
from .data import load_cases
from .logging_setup import Tee
from .evaluation.results_report import build_approach1_results_html
from .orchestration import run_one_config, save_aggregate_summary
from .splits import build_run_configs
from common.llm_models import DEFAULT_MODEL, normalize_model_name
from common.reasoning_effort import DEFAULT_REASONING_EFFORT, REASONING_EFFORT_CHOICES


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureGPT 3-tier dispersion/relapse pipeline + evaluation")
    parser.add_argument("--csv-path", "-c", default=config.CSV_PATH, help="Path to input CSV file.")
    parser.add_argument("--outdir", "-o", default=config.OUT_DIR, help="Root directory to write all shotset/tier outputs.")
    parser.add_argument("--env-path", default=config.ENV_PATH, help="Path to .env containing model-specific API keys.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["gpt-5-nano", "gpt-5"],
        help="LLM deployment to use (gpt-5-nano uses SANDBOX_API_KEY; gpt-5 uses NEW_SECUREGPT_API_KEY).",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Deprecated alias for --model.",
    )
    parser.add_argument("--api-version", default=config.API_VERSION, help="Azure OpenAI API version.")
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=list(REASONING_EFFORT_CHOICES),
        help="GPT-5 reasoning effort sent to the API (default: medium; use 'none' to omit).",
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
    selected_model = normalize_model_name(args.deployment or args.model)

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
            print(f"[START] MODEL={selected_model}")
            print(f"[START] DEPLOYMENT={selected_model}")
            print(f"[START] API_VERSION={args.api_version}")
            print(f"[START] REASONING_EFFORT={args.reasoning_effort}")
            print(f"[START] LOG_PATH={log_path}")
            print(f"[START] RESUME={args.resume}")
            print(f"[START] SKIP_COMPLETED_CONFIGS={args.skip_completed_configs}")
            print(f"[START] FORCE_RERUN_CASES={args.force_rerun_cases}")
            print("=" * 80)

            configure_api(args.env_path, selected_model, args.api_version, reasoning_effort=args.reasoning_effort)
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
                pipeline_cost = aggregate_pipeline_cost_report(run_configs)
                save_pipeline_cost_report(args.outdir, run_configs)
                print_cumulative_report_snapshot(
                    pipeline_cost,
                    label="full pipeline (all shotsets/tiers, resume-stable)",
                )

            save_aggregate_summary(args.outdir, aggregate_summaries)
            build_approach1_results_html(args.outdir)
            final_cost = aggregate_pipeline_cost_report(run_configs)
            save_pipeline_cost_report(args.outdir, run_configs)
            print_cumulative_report_snapshot(
                final_cost,
                label="full pipeline (all shotsets/tiers, final)",
            )
            print("[END] All shotset/tier runs complete.")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
