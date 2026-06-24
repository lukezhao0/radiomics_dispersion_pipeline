"""Public extraction-layer API re-exported for nested evaluation."""

from __future__ import annotations

from ..api.client import preflight_check
from ..api.cost import (
    build_chat_messages,
    configure_global_api_concurrency,
    confirm_cost_estimate_or_exit,
    estimate_prompt_tokens_from_messages,
    print_apriori_cost_estimate_report,
    print_cumulative_report,
    summarize_apriori_cost_estimate,
    write_cost_tracker_json,
)
from ..html_report import (
    build_html_report,
    df_to_html_table,
    html_paragraph,
    html_plot_block,
    html_section,
)
from ..logging_setup import Tee
from ..prompts.builder import build_user_prompt
from .config import MAX_TOKENS
from .data import (
    Case,
    _is_missing_text,
    _selected_report_text,
    _true_dispersion_high_low,
    load_cases,
    make_case_from_row,
)
from .pipeline import extract_subset_records, write_extractions

__all__ = [
    "MAX_TOKENS",
    "Tee",
    "Case",
    "build_chat_messages",
    "build_html_report",
    "build_user_prompt",
    "configure_global_api_concurrency",
    "confirm_cost_estimate_or_exit",
    "df_to_html_table",
    "estimate_prompt_tokens_from_messages",
    "extract_subset_records",
    "html_paragraph",
    "html_plot_block",
    "html_section",
    "load_cases",
    "make_case_from_row",
    "preflight_check",
    "print_apriori_cost_estimate_report",
    "print_cumulative_report",
    "summarize_apriori_cost_estimate",
    "write_cost_tracker_json",
    "write_extractions",
    "_is_missing_text",
    "_selected_report_text",
    "_true_dispersion_high_low",
]
