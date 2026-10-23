# 0011: Pre-specified headline models for the combined-modality audit

Status: Accepted (confirmed by the project owner on 2026-10-22: chosen before seeing results)

## Context
Approach 2 produces dozens of dataset × representation × model results.
Reporting the best one on the same held-out predictions is optimistic.

## Decision
`metrics_audit/compute_unified_metrics.py` fixes the Approach 2 headline as
`dataset_key = combined`, `representation = group_count`, with:
- `pls_regression` for continuous dispersion
- `linear_svm` for high/low
- `ridge_logistic` for relapse

It also fixes the Approach 1 headline as `shotset_high_0_2_low_101_102 / mri_plus_pathology`.
These were chosen before results were seen.

## Consequences
- The audit's comparison is not subject to best-of selection bias.
- The automated Approach 2 reports (`_best_row`) and `experiment_comparison` "best" metrics *are* best-of selections and should be labeled exploratory.
- Changing the headline models after seeing results would undo this property. Record any change as a new ADR.
