# Combined-modality metrics audit (Approach 1 vs Approach 2)

Offline audit of **fused MRI + pathology** performance for tumor dispersion and relapse prediction. Reads saved prediction artifacts only — no LLM re-runs.

**Scope:** All metrics documented here are **combined modality only**.

| Approach | Modality filter | Source |
| --- | --- | --- |
| Approach 1 (few-shot LLM) | `mri_plus_pathology` | `shotset_high_0_2_low_101_102/mri_plus_pathology/predictions_testing_cases.csv` |
| Approach 2 (feature discovery + ML) | `dataset_key == "combined"` | `nested_outer_predictions_case_deduplicated.csv` |

MRI-only, pathology-only, and pathology-calibrated MRI tiers exist in both result directories but are **excluded** from this audit.

## Result directories (June 2026 runs)

- **Approach 1:** `sabcs/securegpt_dispersion_approach1_pipeline_062726`
- **Approach 2:** `sabcs/securegpt_dispersion_approach2_pipeline_062726`
- **Audit outputs:** `sabcs/metrics_audit_combined_modality/`

## Quick start

From the repository root:

```bash
MPLCONFIGDIR=/tmp/mplcache .venv/bin/python pipeline/metrics_audit/compute_unified_metrics.py \
  --approach1-dir sabcs/securegpt_dispersion_approach1_pipeline_062726 \
  --approach2-dir sabcs/securegpt_dispersion_approach2_pipeline_062726 \
  --outdir sabcs/metrics_audit_combined_modality
```

Outputs:

- `unified_metrics_long.csv` — long-format metrics with point estimates and 95% CIs
- `unified_metrics_report.md` — wide comparison table (auto-generated from the CSV)

Bootstrap defaults: **B = 1000**, **seed = 42**, case-level percentile CIs (2.5th / 97.5th percentiles).

## Source artifacts

### Approach 1 — `mri_plus_pathology`

| File | Role |
| --- | --- |
| `predictions_testing_cases.csv` | Per-case predictions (N = 82) |
| `evaluation_metrics_summary.json` | Point metrics |
| `bootstrap_metric_cis.json` / `.csv` | Precomputed 95% CIs |
| `evaluation_metrics_from_csv.txt` | Metrics recomputed from CSV (verification) |
| `run_config.json` | Split design: 4 few-shot rows excluded; 18 MRI-missing rows skipped |

Every row in the predictions CSV has `modality=mri_plus_pathology`, `has_preop_mri=True`, and `has_path_report=True`.

**Evaluation design:** Fixed held-out test after excluding few-shot exemplars (rows 0, 2, 101, 102) and cases without MRI reports.

### Approach 2 — `combined`

| File | Role |
| --- | --- |
| `nested_outer_predictions_case_deduplicated.csv` | One held-out prediction per case (N = 60 for combined) |
| `nested_outer_predictions_all.csv` | Raw per-split predictions |
| `nested_outer_metrics_summary.csv` | Aggregated metrics + bootstrap CIs |
| `nested_resampling_summary.txt` | Headline model selection |
| `feature_density_summary_by_modality.csv` | Lexicon discovery counts |

**Headline combined models** (`group_count` representation):

- Continuous dispersion → `pls_regression`
- High/low dispersion → `linear_svm`
- Relapse → `ridge_logistic`

**Evaluation design:** Nested outer resampling (5 repeated Monte Carlo outer splits). Feature discovery and model fitting occur within outer-train folds only; metrics aggregate case-deduplicated outer-test predictions.

**Feature discovery (not model input dimension):** 1,666 stable phrases (566 MRI + 1,100 pathology) and 320 stable groups (156 MRI + 164 pathology). Headline `group_count` models use ~34 selected features per split after inner CV.

## Unified metric definitions

High/low dispersion is defined as **dispersion score ≥ 85** (`DISPERSION_HIGH_THRESHOLD` in `approach1/config.py`).

| Task | Primary metrics | Notes |
| --- | --- | --- |
| Continuous dispersion | Spearman ρ, MAE (+ 95% CI) | Also RMSE and Pearson r in long CSV |
| High/low dispersion | AUROC, AUPRC, F1, sensitivity, specificity | Report discrimination and threshold-dependent metrics; do not rely on accuracy alone |
| Relapse | AUROC, AUPRC, F1, sensitivity, specificity | A1 relapse AUROC uses binary `relapse_pred` (0/1), not a probability score |

Reporting aligns with common clinical ML conventions (TRIPOD-style): correlation + absolute error for continuous outcomes; AUROC + **AUPRC** for imbalanced binary tasks; bootstrap CIs preferred over p-values.

## Headline results (combined MRI + pathology)

Values are point estimates with 95% bootstrap CIs in parentheses. Recomputed from saved predictions (see `unified_metrics_long.csv`).

| Task | Approach 1 (N = 82) | Approach 2 (N = 60) |
| --- | --- | --- |
| **Continuous dispersion** | Spearman ρ = 0.738 (0.601–0.838); MAE = 51.2 (39.7–62.8) | Spearman ρ = 0.493 (0.253–0.683); MAE = 72.8 (55.3–90.7) |
| **High/low dispersion** | AUROC = 0.868 (0.782–0.941); AUPRC = 0.836 (0.712–0.930); F1 = 0.730; prev 44% | AUROC = 0.752 (0.621–0.866); AUPRC = 0.746 (0.598–0.878); F1 = 0.642; prev 50% |
| **Relapse** | AUROC = 0.668 (0.531–0.812)\*; AUPRC = 0.264 (0.127–0.447); F1 = 0.421; prev 17% | AUROC = 0.898 (0.803–0.967); AUPRC = 0.651 (0.366–0.882); F1 = 0.552; prev 17% |

\*Approach 1 relapse AUROC ranks the binary LLM label, not a continuous risk score.

## Comparability between approaches

**Verdict: approximately comparable, not directly comparable.**

| Factor | Status |
| --- | --- |
| Combined MRI + pathology text | Yes — both approaches |
| High/low threshold (≥ 85) | Yes |
| Relapse label definition | Yes (similar prevalence where cases overlap) |
| MRI-missing exclusion (18 cases) | Yes |
| Identical evaluated cases | **No** — 56-case overlap; A1 N = 82, A2 N = 60 |
| Split protocol | **No** — A1 fixed holdout; A2 nested CV |
| Few-shot cases in test | A1 excludes rows 0, 2, 101, 102; **A2 includes them** |
| Methodology | A1 direct LLM prediction; A2 LLM feature extraction → supervised ML |

**Subset note:** 26 cases evaluated by Approach 1 lack Approach 2 combined predictions (likely incomplete combined feature extraction). The A2 subset is more balanced for high/low dispersion (50% vs 44% in A1). Avoid direct superiority claims; paired analysis on the 56-case overlap is more appropriate for exploratory comparison.

## Interpretation summary

**Strongest**

- Approach 1 — continuous dispersion and high/low classification (larger N, narrower CIs).
- Approach 2 — relapse discrimination numerically (uses probability scores; AUROC/AUPRC higher).

**Weakest**

- Approach 1 — relapse (low AUPRC; ~14 events; accuracy misleading under 17% prevalence).
- Approach 2 — continuous dispersion (Spearman ρ ≈ 0.49 on a smaller subset).

**Cautions for medical reporting**

- Describe as **internal validation**, not clinical validation or deployment readiness.
- Relapse CIs are wide (e.g., A2 sensitivity 0.50–1.00); event counts are small.
- Approach 2 relapse metrics may reflect optimism from nested model/representation selection.
- Accuracy alone is insufficient for relapse; emphasize AUPRC and sensitivity/specificity.

## Reviewer concerns (phrasing guide)

| Concern | Severity | Suggested phrasing |
| --- | --- | --- |
| Non-identical evaluation cohorts | High | "partially overlapping internal validation cohorts" |
| Small relapse event count | High | "exploratory / hypothesis-generating" |
| No external validation | High | "internal validation only" |
| A1 relapse AUROC from binary labels | Moderate | "discrimination based on binary LLM labels" |
| A2 nested model selection | Moderate | "nested cross-validated feature discovery and model fitting" |
| Arbitrary high/low threshold | Moderate | cite threshold explicitly; justify clinically if possible |
| A2 subset selection bias | Moderate | note N = 60 vs N = 82 and missing-case mechanism |

## Script reference

`compute_unified_metrics.py` filters inputs as follows:

```python
# Approach 1
A1_PRED_REL = "shotset_high_0_2_low_101_102/mri_plus_pathology/predictions_testing_cases.csv"

# Approach 2
A2_DATASET = "combined"
A2_REPRESENTATION = "group_count"
A2_HEADLINE_MODELS = {
    "continuous_dispersion": "pls_regression",
    "high_low_dispersion": "linear_svm",
    "relapse": "ridge_logistic",
}
```

Point estimates are recomputed from prediction files. Approach 2 CIs are verified against `nested_outer_metrics_summary.csv` when available.

## Related documentation

- [`../README.md`](../README.md) — pipeline overview and quick start
- [`../experiment_comparison/README.md`](../experiment_comparison/README.md) — config-driven cross-run comparison
- [`../../sabcs/metrics_audit_combined_modality/unified_metrics_report.md`](../../sabcs/metrics_audit_combined_modality/unified_metrics_report.md) — latest generated comparison table
