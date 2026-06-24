"""Runtime symbol resolution for refactored approach2 modules."""

from __future__ import annotations


def test_recoding_recode_has_ensure_case_id_bound() -> None:
    import approach2.recoding as recoding_mod

    assert hasattr(recoding_mod, "ensure_case_id")


def test_audit_compute_mri_audit_has_ensure_case_id_bound() -> None:
    import approach2.audit as audit_mod

    assert hasattr(audit_mod, "ensure_case_id")


def test_reports_regenerate_helpers_imported() -> None:
    import approach2.reports as reports_mod

    assert hasattr(reports_mod, "_save_figure")
    assert hasattr(reports_mod, "_metric_bar_figure_size")
    assert hasattr(reports_mod, "_raw_df_with_row_index")


def test_extraction_pipeline_preflight_imported() -> None:
    import approach2.extraction.pipeline as pipeline_mod

    assert hasattr(pipeline_mod, "preflight_check")
