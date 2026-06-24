"""Standalone extraction CLI."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from datetime import datetime
from typing import Optional

from ..logging_setup import Tee
from common.llm_models import SUPPORTED_MODELS
from common.reasoning_effort import DEFAULT_REASONING_EFFORT, REASONING_EFFORT_CHOICES
from .config import CSV_PATH, DEFAULT_MODEL, DEFAULT_REPORT_MODE, ENV_PATH, REPORT_CONFIG, configure_llm
from .pipeline import _load_row_indices_json, run_pipeline

def main(
    csv_path: str = CSV_PATH,
    out_dir: Optional[str] = None,
    report_mode: str = DEFAULT_REPORT_MODE,
    row_indices_json: Optional[str] = None,
    split_id: Optional[str] = None,
    split_role: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    max_api_workers: Optional[int] = None,
    resume: bool = True,
    force_reextract: bool = False,
    assume_yes: bool = False,
) -> None:
    if report_mode not in REPORT_CONFIG:
        raise ValueError(f"--report-mode must be one of {sorted(REPORT_CONFIG)}")

    if out_dir is None:
        out_dir = os.path.join(
            os.getcwd(),
            f"securegpt_dispersion_feature_discovery_{REPORT_CONFIG[report_mode]['outdir_suffix']}",
        )

    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_suffix = f"_{filename_prefix}" if filename_prefix else ""
    log_path = os.path.join(log_dir, f"run_log_feature_discovery_{report_mode}{log_suffix}.txt")
    row_indices = _load_row_indices_json(row_indices_json)

    log_mode = "a" if resume and os.path.exists(log_path) else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        tee_out = Tee(sys.__stdout__, log_f)
        tee_err = Tee(sys.__stderr__, log_f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[LOG] Logging stdout/stderr to: {log_path}")
            if row_indices is not None:
                print(f"[LOG] Loaded row subset from {row_indices_json} with n={len(row_indices)}")
            run_pipeline(
                csv_path=csv_path,
                out_dir=out_dir,
                report_mode=report_mode,
                resume=resume,
                force_reextract=force_reextract,
                row_indices=row_indices,
                split_id=split_id,
                split_role=split_role,
                filename_prefix=filename_prefix,
                max_api_workers=max_api_workers,
                assume_yes=assume_yes,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SecureGPT modality-specific lexical feature discovery pipeline"
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=DEFAULT_MODEL,
        choices=list(SUPPORTED_MODELS),
        help="LLM deployment to use (gpt-5-nano uses SANDBOX_API_KEY; gpt-5 uses NEW_SECUREGPT_API_KEY).",
    )
    parser.add_argument(
        "--env-path",
        dest="env_path",
        default=ENV_PATH,
        help="Path to .env containing model-specific API keys.",
    )
    parser.add_argument(
        "--reasoning-effort",
        dest="reasoning_effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=list(REASONING_EFFORT_CHOICES),
        help="GPT-5 reasoning effort sent to the API (default: minimal; use 'none' to omit).",
    )
    parser.add_argument(
        "--csv-path",
        "-c",
        dest="csv_path",
        default=CSV_PATH,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        dest="out_dir",
        default=None,
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--report-mode",
        "-m",
        dest="report_mode",
        choices=sorted(REPORT_CONFIG.keys()),
        default=DEFAULT_REPORT_MODE,
        help="Which modality to extract from: mri or path.",
    )
    parser.add_argument(
        "--row-indices-json",
        dest="row_indices_json",
        default=None,
        help="Optional JSON file containing a list of integer row indices to extract.",
    )
    parser.add_argument(
        "--split-id",
        dest="split_id",
        default=None,
        help="Optional outer split identifier recorded in the outputs.",
    )
    parser.add_argument(
        "--split-role",
        dest="split_role",
        default=None,
        help="Optional split role recorded in the outputs, e.g. train/test.",
    )
    parser.add_argument(
        "--filename-prefix",
        dest="filename_prefix",
        default=None,
        help="Optional filename prefix for outputs written to outdir.",
    )
    parser.add_argument(
        "--max-api-workers",
        dest="max_api_workers",
        type=int,
        default=None,
        help="Maximum number of concurrent API extraction workers. Defaults to a conservative hardware-aware value.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=False,
        help="Accept the printed a-priori LLM cost estimate and skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable reuse of per-case extraction checkpoints.",
    )
    parser.add_argument(
        "--force-reextract",
        dest="force_reextract",
        action="store_true",
        default=False,
        help="Ignore existing per-case checkpoints and call the API again.",
    )
    args = parser.parse_args()
    configure_llm(args.model, env_path=args.env_path, reasoning_effort=args.reasoning_effort)
    main(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        report_mode=args.report_mode,
        row_indices_json=args.row_indices_json,
        split_id=args.split_id,
        split_role=args.split_role,
        filename_prefix=args.filename_prefix,
        max_api_workers=args.max_api_workers,
        resume=args.resume,
        force_reextract=args.force_reextract,
        assume_yes=args.yes,
    )
