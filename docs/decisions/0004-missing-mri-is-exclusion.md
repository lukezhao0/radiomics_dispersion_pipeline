# 0004: Missing MRI is an exclusion, never a zero vector

Status: Accepted (`approach1_progress.md` / `approach2_progress.md`, 2026-06-23 "Missing MRI handling")

## Context
Some cases have no usable pre-op MRI report (empty or a placeholder such as
`NA`). Encoding them as all-zero features or "no findings" would teach a
model that a missing report means a clean MRI.

## Decision
- MRI-derived pathways drop these cases explicitly and log counts: Approach 1 tiers `mri_only` and `mri_plus_pathology`; Approach 2 datasets `mri`, `combined`, `mri_pathcal*` and `mri_teacher_student`.
- Pathology-only pathways keep them.
- Approach 2 also skips MRI extraction API calls for them.

## Consequences
- MRI and pathology results are on different case sets. Approach 2 adds `pathology_only_mri_complete_subset_metrics.csv` for a matched comparison; Approach 1 has no equivalent.
- Exclusion counts are written to `skipped_cases_missing_mri.csv` (A1) and `mri_missing_case_summary.csv` (A2).
