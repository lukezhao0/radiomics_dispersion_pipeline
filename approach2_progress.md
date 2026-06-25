# Clinical NLP Dispersion Pipeline Change Log

This Markdown file tracks major implementation changes to the leakage-aware clinical NLP and machine learning pipeline for predicting breast tumor dispersiveness from MRI and pathology reports. It is intended to be extended over time with new dated entries as additional features, fixes, and methodological updates are added.

---

## 2026-06-22 — GPT-5-nano Global API Migration, Cost Tracking, and Pre-Run Cost Estimation

### Summary

Updated the two-file pipeline implementation to replace the outdated SecureGPT/AzureOpenAI API usage with the current Stanford AI Sandbox API pattern for **GPT-5-nano Global**, while adding both post-usage and a-priori cost estimation. The changes were intentionally limited to the API layer, LLM usage accounting, prompt layout for caching, and the main pipeline preflight flow.

### Files Modified

- `feature_discovery_eval_ML_aux2.py`
  - Auxiliary extraction/API helper file.
  - Centralized LLM call logic, prompt construction, usage accounting, and cost reporting.

- `feature_discovery_eval_ML.py`
  - Main nested evaluation driver.
  - Imports the new estimation/reporting utilities and runs a pre-pipeline cost estimate before extraction begins.

### API Migration

Replaced the older API client configuration with direct `requests.post(...)` calls following the working Stanford AI Sandbox example.

New API configuration:

```python
DEPLOYMENT = "gpt-5-nano"
API_VERSION = "2024-12-01-preview"

URL = (
    "https://aihubapi.stanfordhealthcare.org/azure-openai"
    f"/deployments/{DEPLOYMENT}/chat/completions"
    f"?api-version={API_VERSION}"
)

HEADERS = {
    "api-key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "application/json",
}
```

Key changes:

- Uses `SANDBOX_API_KEY` instead of the old `SECUREGPT_API_KEY`.
- Uses the `api-key` header rather than `Ocp-Apim-Subscription-Key`.
- Uses direct HTTP requests instead of the older `AzureOpenAI` client object.
- Uses `max_completion_tokens`, appropriate for GPT-5-style deployments.
- Keeps robust API error printing for non-OK responses.
- Preserves warning behavior when the model returns empty visible output, especially when `finish_reason="length"`.

### Pricing Assumptions Added

Added editable pricing constants for **GPT-5-nano Global**:

```python
PRICE_PER_1M_INPUT_TOKENS = 0.05
PRICE_PER_1M_CACHED_INPUT_TOKENS = 0.01
PRICE_PER_1M_OUTPUT_TOKENS = 0.40
```

Billing interpretation:

- Uncached input tokens are billed at the input-token rate.
- Cached input tokens are billed at the cached-input-token rate.
- Completion tokens are billed at the output-token rate.
- Completion tokens include both visible output tokens and reasoning tokens when reported by the API.

### Post-Usage Cost Tracking

Implemented a cumulative LLM usage and cost tracker in the auxiliary file.

Tracked fields include:

- Number of LLM calls.
- Prompt tokens.
- Cached prompt tokens.
- Uncached prompt tokens.
- Completion tokens.
- Reasoning tokens.
- Total tokens.
- Estimated total cost in USD.
- Estimated cache savings in USD.

The usage parser reads token accounting from the API response, including:

```python
usage["prompt_tokens"]
usage["completion_tokens"]
usage["total_tokens"]
usage["prompt_tokens_details"]["cached_tokens"]
usage["completion_tokens_details"]["reasoning_tokens"]
```

The pipeline now computes:

- Actual estimated input cost.
- Actual estimated cached-input cost.
- Actual estimated output cost.
- Total estimated call cost.
- Cache savings relative to billing all prompt tokens at the uncached input-token rate.

### Cumulative Cost Report Output

Added reporting utilities that print a cumulative token/cost report and write the final report to disk.

Expected final JSON output:

```text
llm_token_cost_report.json
```

The report captures both aggregate token usage and estimated dollar cost after the run completes.

### A-Priori Cost Estimation Before Full Pipeline Run

Added a pre-run estimation step in the main evaluation script.

Before calling the full extraction workflow, the pipeline now:

1. Loads the dataset.
2. Constructs the planned outer train/test splits.
3. Builds the actual LLM prompts that would be sent for the planned extraction calls.
4. Estimates total input token volume from those prompts.
5. Estimates full-run output token volume using the configured maximum completion token budget.
6. Computes a conservative no-cache cost estimate.
7. Computes a cache-aware cost estimate based on repeated stable prompt-prefix tokens.
8. Prints the estimate to the command line.
9. Requires explicit permission before continuing.

The user must type:

```text
YES
```

to proceed with the full pipeline.

A non-interactive override was also added:

```bash
--yes
```

This allows batch execution after the cost estimate behavior has been reviewed.

### Cache-Aware Prompt Adjustment

Adjusted prompt construction to improve prompt-prefix caching across cases.

Main prompt-layout change:

- Stable task instructions, schema requirements, ontology guidance, and extraction rules are placed before case-specific content.
- Case-specific fields such as `case_id`, `index_side`, selected report metadata, and report text are placed after the reusable stable prefix.

Rationale:

- API-side prompt caching is most effective when repeated requests share the same leading prefix.
- By moving static instructions earlier and report-specific content later, repeated extraction calls are more likely to share cached input tokens.

### Cache Savings Tracking

Added explicit cached-token accounting and dollar savings estimation.

For each API response:

```python
cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)
```

Cache savings are estimated as:

```text
cost if all prompt tokens were uncached - actual cached/uncached input cost
```

This makes it possible to monitor whether the prompt restructuring is producing meaningful savings across repeated case-level calls.

### Main Driver Integration

The main evaluator now imports additional utilities from the auxiliary file, including pre-run estimation and final cost-report writing functions.

The pipeline flow is now conceptually:

1. Parse CLI arguments.
2. Load cases and target labels.
3. Build planned outer splits.
4. Estimate LLM cost for the planned extraction workload.
5. Ask for command-line confirmation unless `--yes` is supplied.
6. Run the existing preflight checks.
7. Continue with the original leakage-aware extraction, feature discovery, modeling, and evaluation workflow.
8. Write the cumulative token/cost report at the end.

### Validation Performed

Basic validation was performed after editing:

- Both edited Python files passed syntax compilation.
- The edited files were imported together using a dummy `SANDBOX_API_KEY`.
- The updated API path was not live-tested because that would require a real Stanford Sandbox key and would incur usage.

### Example Usage

Interactive run with pre-run estimate and manual confirmation:

```bash
python feature_discovery_eval_ML.py \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/securegpt_dispersion_pathology_informed_eval \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined
```

Non-interactive run after reviewing the estimation behavior:

```bash
python feature_discovery_eval_ML.py \
  --csv-path /Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/securegpt_dispersion_pathology_informed_eval \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined \
  --yes
```

### Notes and Caveats

- Cost estimates are approximate and depend on the API-reported token accounting.
- A-priori estimates use approximate local token estimation rather than the model provider’s exact tokenizer.
- The conservative estimate assumes maximum completion-token usage for every call, so actual cost may be lower if model responses terminate earlier.
- Cache-aware estimates are approximate before the run; actual cache savings should be evaluated from the post-run usage report.
- The update does not change the nested leakage-aware modeling design, ontology logic, feature discovery strategy, or downstream ML model families beyond what was necessary for API usage and cost accounting.

### Future Change Log Template

Use this template for future additions:

```markdown
## YYYY-MM-DD — Short Feature or Fix Title

### Summary

Briefly describe the change and why it was added.

### Files Modified

- `file_name.py`
  - Short description of the edits.

### Implementation Details

- Detail 1.
- Detail 2.
- Detail 3.

### Validation Performed

- Test or check 1.
- Test or check 2.

### Notes and Caveats

- Limitation 1.
- Follow-up 1.
```

---

## 2026-06-22 — Missing-MRI Handling, Relapse Prediction, Resume Support, and Persistent Logs

### Summary

Updated the two-file nested clinical NLP + ML pipeline to add the requested missing-MRI safeguards, relapse-status modeling, relapse metrics/plots, resume behavior, and persistent logging. Edits were intentionally limited to the main evaluator and auxiliary extraction/logging helper.

### Files Modified

- `feature_discovery_eval_ML.py`
  - Main nested evaluation driver.
  - Adds MRI-missing filtering for MRI-derived evaluations.
  - Adds relapse-status classification using the same extracted dispersion-vector representations.
  - Adds relapse-specific metrics, bootstrap confidence intervals, plots, and curve outputs.
  - Adds target-aware prediction, metric, coefficient, and feature-ranking outputs.
  - Saves logs under `logs/` inside the run output directory.

- `feature_discovery_eval_ML_aux2.py`
  - Auxiliary extraction/API helper file.
  - Keeps per-case extraction checkpoints and resume behavior.
  - Moves standalone extraction logs under `logs/` inside the extraction output directory.

### Missing MRI Handling

The pipeline now explicitly identifies cases with missing `preop_MRI_text`. These cases are retained for pathology-only evaluations because every case is expected to have a pathology report, but are skipped for evaluations whose feature representation requires MRI text:

- MRI-only datasets: `mri`
- MRI + pathology early-fusion datasets: `combined`
- Pathology-calibrated weighted MRI datasets: `mri_pathcal_weighted` and calibration ablations
- Teacher-student MRI datasets: `mri_teacher_student`

This prevents missing-MRI cases from being represented as all-zero MRI feature vectors, which would otherwise create misleading MRI-only or combined-model training and held-out predictions.

### Relapse Prediction

The classical supervised ML stage now trains classification models for two binary targets:

1. `dispersion_high_low`, using `dispersion_true_high_low`
2. `relapse_status`, using `relapse_true`

Continuous regression models continue to predict only:

- `dispersion_score`, using `dispersion_true`

Predictions now include `target_name` and `target_col`, allowing downstream outputs to distinguish dispersion high/low classification from relapse classification.

### Relapse Metrics Added

For binary classification, including relapse prediction, the pipeline now reports:

- AUROC
- AUPRC
- Brier score
- Accuracy
- F1
- Precision
- Recall / sensitivity
- Specificity
- True negatives, false positives, false negatives, true positives
- Prevalence
- Positive prediction rate
- Calibration intercept
- Calibration slope
- Bootstrap confidence intervals for aggregate held-out metrics

The final aggregate metrics file remains:

```text
nested_outer_metrics_summary.csv
```

Per-split metrics remain:

```text
nested_outer_fold_metrics_all.csv
```

### Relapse Plots Added

The pipeline now writes relapse-specific visualization files when relapse predictions are available:

```text
nested_relapse_auroc_comparison.png
nested_relapse_auprc_comparison.png
nested_relapse_f1_comparison.png
nested_relapse_brier_comparison.png
nested_relapse_roc_curves_top_models.png
nested_relapse_pr_curves_top_models.png
```

The existing classification comparison plot is now target-aware, so it can include both dispersion high/low and relapse classification results.

### Resume and Checkpoint Behavior

The pipeline continues to support two complementary resume layers:

1. **Per-case extraction checkpoints**
   - Stored under each modality/split checkpoint folder.
   - Prevents repeated LLM calls for already completed case-level extractions.

2. **Completed split checkpoints**
   - Stored under each split’s `_split_resume_checkpoint/` directory.
   - Allows a rerun to skip completed outer splits and reload their saved predictions, metrics, hyperparameters, coefficients, lexicons, audit tables, and calibration tables.

Relevant flags:

```bash
--no-resume
--force-reextract
--no-skip-completed-splits
```

Default behavior is resume-enabled.

### Persistent Logs

Console output and errors are now saved to log files under a `logs/` folder:

```text
<out_dir>/logs/run_log_feature_discovery_nested_eval.txt
<out_dir>/logs/run_log_feature_discovery_<report_mode>[_prefix].txt
```

When resume is enabled and a log already exists, new output is appended rather than overwriting the prior log.

### Ablation Explanation

In this codebase, an ablation is a deliberately altered control analysis that removes, randomizes, or breaks one part of the proposed method while keeping the rest of the pipeline similar. The goal is to test whether a specific component is actually responsible for performance.

For example, the pathology-calibrated MRI pathway learns MRI concept weights using training-only MRI-pathology concordance. The randomized-pathology and mismatched-pairing ablations test whether any observed improvement truly depends on biologically meaningful MRI-pathology alignment. If the ablated versions perform similarly to the real calibration, the apparent benefit may reflect nonspecific regularization or overfitting rather than true pathology-informed signal.

### Validation Performed

- Both edited Python files passed syntax compilation with `python -m py_compile`.
- Both edited files successfully displayed CLI help with a dummy `SANDBOX_API_KEY`.
- Core relapse metric, bootstrap-CI, and missing-MRI filtering helpers were smoke-tested on synthetic data.

### Notes and Caveats

- Relapse prediction is implemented as binary classification from the same extracted lexical dispersion-vector representations.
- AUROC and AUPRC are only meaningful when both relapse classes are present in the evaluated held-out predictions.
- Small relapse-positive counts may make fold-level relapse metrics unstable; bootstrap intervals are intended to make this uncertainty visible, not eliminate it.
- MRI-missing cases are still retained in pathology-only analyses.
- External validation remains necessary before making clinical claims.

---

## Publication-Quality Follow-up Suggestions

To make the analysis stronger for an abstract or manuscript, prioritize:

1. Report the number of MRI-missing cases and how many were skipped per modality and split.
2. Pre-specify the relapse endpoint definition and follow-up window.
3. Report relapse prevalence and class counts per outer split.
4. Compare relapse prediction from lexical vectors against simple clinical baselines, such as residual cancer burden, stage, receptor status, pathologic complete response, nodal status, or tumor size, if available.
5. Add decision-curve analysis or net-benefit plots if the goal is clinical risk stratification.
6. Add permutation testing for relapse AUROC/AUPRC to quantify whether performance exceeds chance under label shuffling.
7. Add calibration plots for relapse probabilities.
8. Add nested feature-stability summaries specifically for relapse coefficients.
9. Include representative correctly predicted and missed relapse cases with quote-grounded lexical evidence.
10. Validate on an external cohort or at minimum use a locked final model on a held-out validation set not used in any feature discovery.

## 2026-06-22 — Fold-Level Parallelism, Automated Reports, Interpretability, Missed-Case Analysis, and Relapse Diagnostics

### Summary

Updated the two-file nested clinical NLP + ML pipeline to implement the requested concurrency hardening, fold-level parallel execution, automated results reporting, interpretability reporting, missed-case/error analysis, and relapse-specific statistical diagnostics. The implementation preserves the leakage-aware design by keeping every outer fold isolated in its own output directory with fold-specific train/test membership, frozen lexicons, calibration artifacts, model outputs, checkpoints, and logs.

### Files Modified

- `feature_discovery_eval_ML.py`
  - Main nested evaluation driver.
  - Adds `--parallel-fold-workers` to run outer folds concurrently when requested.
  - Coordinates fold, modality, API, and ML worker settings to reduce oversubscription.
  - Adds split provenance manifests and per-fold failure capture.
  - Adds one-prediction-per-case deduplication before aggregate metric computation.
  - Adds automated Markdown/HTML performance reports, interpretability reports, missed-case reports, relapse class-balance diagnostics, and relapse permutation tests.
  - Expands binary metrics to include balanced accuracy, PPV/precision, NPV, and no-skill AUPRC baseline.

- `feature_discovery_eval_ML_aux2.py`
  - Auxiliary extraction/API helper file.
  - Adds a process-wide Stanford AI Sandbox API semaphore so `--max-api-workers` caps active HTTP requests globally across folds, modalities, and case-level extraction threads.
  - Logs the global API cap during extraction.
  - Keeps existing per-case extraction checkpoints and resume behavior.

### Concurrency and Parallelism Audit

The prior implementation already supported case-level LLM extraction parallelism inside `extract_subset_records()` through `ThreadPoolExecutor`, modality-level processing across MRI and pathology within each outer split through `--parallel-modality-workers`, and ML model fitting parallelism inside `GridSearchCV` through `--ml-n-jobs`. However, the outer CV folds themselves were still run sequentially. The updated implementation adds a fold-level orchestration layer so outer folds can be run concurrently with `--parallel-fold-workers`, including the intended 5-fold case of:

```bash
--outer-scheme stratified_kfold \
--outer-folds 5 \
--parallel-fold-workers 5
```

Each outer fold is now executed by `run_one_outer_split()`, which uses only split-local result containers and writes only under:

```text
<out_dir>/outer_splits/<split_id>/
```

This avoids concurrent mutation of aggregate result lists. The main thread collects completed fold results and concatenates them only after each fold worker returns.

### Leakage-Aware Fold Isolation

Each fold now writes immutable split provenance files:

```text
<split_id>_split_provenance.csv
<split_id>_split_manifest.json
```

The manifest records train/test case IDs, train/test case hashes, split sizes, seed, and outer split settings. Fold-specific outputs remain isolated, including:

- train-only LLM extraction checkpoints
- stable phrase/group lexicons
- frozen ontology recoding outputs
- MRI audit tables
- MRI-pathology reliability matrices
- weighted MRI lexicons
- weighted concept-score matrices
- model predictions, metrics, hyperparameters, and coefficients
- completed split checkpoints
- failed split markers when exceptions occur

### API and CPU Oversubscription Controls

The command-line controls now include:

```bash
--parallel-fold-workers
--max-api-workers
--parallel-modality-workers
--ml-n-jobs
```

`--max-api-workers` is now interpreted as a global active API request cap, not a per-fold/per-modality multiplier. The auxiliary API client enforces this with a process-wide bounded semaphore. This means that even if 5 folds and 2 modalities are running concurrently, the pipeline will not exceed the requested number of simultaneous Stanford AI Sandbox HTTP requests.

The main driver also coordinates local CPU usage by reducing `ml_n_jobs` when the requested product of fold workers, modality workers, and ML workers would obviously exceed available CPU cores.

### Per-Fold Exception Capture and Resume Safety

Each outer fold is wrapped in exception capture. If a fold fails, the pipeline writes:

```text
<out_dir>/outer_splits/<split_id>/_split_resume_checkpoint/FAILED.json
nested_outer_split_errors.csv
```

Other folds can still complete. Completed folds continue to write `_split_resume_checkpoint/COMPLETED.json` plus checkpointed predictions, metrics, hyperparameters, coefficients, lexicons, audit tables, reliability matrices, and weighted lexicons. Resume mode can still skip completed folds and reload their per-fold outputs.

### One-Prediction-Per-Case Aggregate Metrics

Raw per-split predictions are still saved in:

```text
nested_outer_predictions_all.csv
```

A new deduplicated prediction file is also written:

```text
nested_outer_predictions_case_deduplicated.csv
```

For stratified 5-fold CV this should preserve one held-out prediction per case. For repeated Monte Carlo splits, cases may appear in multiple test sets; repeated predictions are averaged into one prediction per case/model/target before computing the final aggregate metrics. The final metrics file now uses this deduplicated prediction layer:

```text
nested_outer_metrics_summary.csv
```

### Automated Results Report

The pipeline now writes:

```text
automated_results_report.md
automated_results_report.html
```

The report summarizes major model pathways when available, including MRI-only, pathology-only, naive MRI+pathology combined, pathology-calibrated MRI, calibration-ablation, and teacher-student models. It reports regression metrics for continuous dispersion score and classification metrics for dispersion high/low and relapse status, including bootstrap confidence intervals when estimable.

Additional report plots are saved under:

```text
report_plots/
```

These include predicted-vs-true plots, residual plots, confusion matrices, ROC curves, precision-recall curves, calibration plots, ranked model-performance plots, and bootstrap confidence-interval plots.

### Interpretability Report

The pipeline now writes:

```text
interpretability_report.md
interpretability_report.html
feature_density_summary_by_modality.csv
top_model_coefficients_interpretability.csv
feature_association_dispersion_vs_relapse.csv
```

The interpretability report summarizes feature density by modality, stable phrase/group summaries, coefficient tables, fold-level coefficient sign stability, pathology-MRI reliability matrices, and calibration-derived MRI concept weights. Coefficient tables are annotated with coefficient sign, training/test feature prevalence, feature modality, and inferred ontology concept when available.

MRI and pathology are intentionally not forced to have exactly equal feature counts. The report documents this explicitly, because equalizing counts by default could suppress real modality-specific information or create artificial comparability. Matched top-k or sparsity-controlled comparisons should be treated as sensitivity analyses.

### Missed-Case and Error Analysis

The pipeline now writes:

```text
missed_case_error_analysis.csv
missed_case_error_analysis.md
```

For regression, cases are ranked by absolute residual, signed residual, and standardized residual, and labeled as strong overpredictions or underpredictions. For classification, the report identifies false positives, false negatives, low-confidence correct predictions, and high-confidence incorrect predictions. The output includes case ID, fold provenance, true label, predicted value/risk, predicted class, confidence, error type, and heuristic failure-mode annotations such as missing MRI report, sparse MRI language, near-threshold dispersion, or low-confidence boundary case.

### Relapse-Specific Diagnostics

Relapse analysis now includes:

```text
relapse_class_balance_by_split.csv
relapse_permutation_tests.csv
```

The class-balance table reports relapse counts and prevalence overall, per split, and within train/test partitions, with warnings when a partition has too few relapse-positive cases for reliable AUROC, AUPRC, calibration, or confidence-interval estimation. The metrics table reports the no-skill AUPRC baseline equal to relapse prevalence.

Relapse permutation tests shuffle relapse labels over held-out case-level predictions and compare observed AUROC/AUPRC against the null distribution to estimate empirical p-values. The number of permutations is controlled by:

```bash
--relapse-permutation-n
```

Use `--relapse-permutation-n 0` to disable this step.

### New Output Files

Core new aggregate outputs include:

```text
nested_outer_predictions_case_deduplicated.csv
nested_outer_split_errors.csv
automated_results_report.md
automated_results_report.html
interpretability_report.md
interpretability_report.html
feature_density_summary_by_modality.csv
top_model_coefficients_interpretability.csv
feature_association_dispersion_vs_relapse.csv
missed_case_error_analysis.csv
missed_case_error_analysis.md
relapse_class_balance_by_split.csv
relapse_permutation_tests.csv
pathology_only_mri_complete_subset_metrics.csv
```

### Validation Performed

- Both edited Python files passed syntax compilation with `python -m py_compile`.
- `feature_discovery_eval_ML.py --help` successfully displayed the new CLI flags with a dummy `SANDBOX_API_KEY`.
- `feature_discovery_eval_ML_aux2.py --help` successfully displayed with a dummy `SANDBOX_API_KEY`.
- Core case-deduplication, metrics, and report-plot helper functions were smoke-tested on synthetic prediction data.

### Notes and Caveats

- Fold-level parallelism is safest when each fold writes to its own output directory, which is now enforced by design.
- `--max-api-workers` caps active API requests globally, but thread pools may still create more waiting worker threads when fold and modality parallelism are both high.
- Matplotlib plot generation remains aggregate-stage only, after fold workers complete, to avoid concurrent plotting from multiple fold threads.
- Relapse permutation testing is performed on aggregate held-out case-level predictions; it tests whether the final score/probability ranking exceeds label-shuffled performance, but it does not create new LLM extractions or retrain all models under each permutation.
- External validation remains necessary before making clinical claims.

---

## 2026-06-23 — Phase-1 Modular Package Extraction (approach2/)

### Summary

Introduced a first-stage modular package under `pipeline/approach2/` by extracting pure, low-coupling helpers from the monolithic `approach2.py` driver. Orchestration, nested CV fitting, reporting, and the full `approach2_aux.py` extraction body remain unchanged in this pass. Scientific behavior (prompts, thresholds, splits, metrics formulas, output schemas) is preserved.

### Files Modified / Added

- `pipeline/approach2.py`
  - Fixed broken import: `approach2_3_aux` → `approach2_aux`.
  - Imports constants, models, text helpers, metric stats, and atomic I/O from the new `approach2/` package.
  - Retains all orchestration, lexicon/rediscovery, calibration, ML fitting, reports, and CLI `main()`.

- `pipeline/approach2_aux.py`
  - Body unchanged; added note that future home is `approach2/extraction/` (phase 2).

- `pipeline/approach2/` (new package)
  - `config.py` — thresholds, ontology, `META_COLS`, pattern constants.
  - `models.py` — `LowInfoFeatureFilter`, `ModelSpec`.
  - `text_utils.py` — normalization, slugging, negation/uncertainty detection, parallelism defaults.
  - `io_atomic.py` — `atomic_write_df`, `safe_read_csv_if_exists`.
  - `metrics/stats.py` — `safe_spearman`, `safe_pearson`, `rmse`, `calibration_intercept_slope`.
  - `__init__.py` — public re-exports.

- `pipeline/pyproject.toml`
  - Added `approach2*` to setuptools package discovery.

- `pipeline/tests/approach2/` (new)
  - Golden/smoke tests for config, text utils, metric stats, imports, and CLI `--help`.

### Import Fix

`approach2.py` previously imported `approach2_3_aux`, which does not exist under `pipeline/`. The import now correctly targets `approach2_aux.py`.

### Intentionally Deferred (not changed in phase 1)

- Full extraction of `approach2_aux.py` into `api/`, `prompts/`, `schema/`, `data/` subpackages.
- Split/lexicon/rediscovery, feature matrices, calibration, orchestration, reports, and CLI extraction from `approach2.py`.
- Consolidation of `_coerce_candidate_concepts` (main uses `make_slug()`; aux uses `.strip()` — different behavior, must stay separate).
- `sabcs/approach2_3*.py` copies outside `pipeline/`.

### Future Refactoring Phases

1. **Phase 2** — Mirror `approach1/` for extraction/API: move `approach2_aux.py` into `approach2/api/`, `prompts/`, `schema/`, `data/`, `html_report/`; thin `approach2_aux.py` shim.
2. **Phase 3** — Extract splits, lexicon, recoding, features, audit, calibration from `approach2.py`.
3. **Phase 4** — Extract ML builders, evaluation metrics/plots/reports.
4. **Phase 5** — Extract `orchestration.py` + `cli.py`; reduce `approach2.py` to a compatibility shim.
5. **Phase 6** — Golden-fold regression tests and leakage checks.

### Validation Performed

```bash
cd pipeline
SANDBOX_API_KEY=dummy python -m py_compile approach2.py approach2_aux.py approach2/*.py
SANDBOX_API_KEY=dummy python approach2.py --help
pytest tests/approach2/
```

All 17 approach2 tests passed.

### Notes and Caveats

- Run scripts from `pipeline/` (or ensure `pipeline/` is on `PYTHONPATH`) so `approach2` package and `approach2_aux` resolve correctly.
- Phase 1 reduces `approach2.py` by ~350 lines without altering nested evaluation logic.
- Deeper modularization should proceed incrementally with golden tests before moving leakage-sensitive split/lexicon code.

---

## 2026-06-23 — Phases 2–5 Modular Package Completion

### Summary

Completed the remaining modularization plan: extracted `approach2_aux.py` into the `approach2` package (extraction/API/prompts/schema/HTML), split the nested evaluation driver into domain modules, and reduced both `approach2.py` and `approach2_aux.py` to thin CLI shims. Scientific behavior, prompts, thresholds, output filenames, and CLI flags are preserved.

### New / Updated Package Layout

```
pipeline/approach2/
  extraction/          # LLM extraction layer (from approach2_aux.py)
    config.py, data.py, text_helpers.py, schema.py, pipeline.py, cli.py
  api/                   # client.py, cost.py
  prompts/               # extraction.py (static), builder.py (dynamic)
  html_report.py
  logging_setup.py
  eval_data.py           # target frames, MRI-missing filters
  splits.py, lexicon.py, recoding.py, audit.py, calibration.py
  features/
    normalize.py, matrices.py
  models_ml.py           # model specs, fitting, teacher-student, metrics
  evaluation/
    plots.py             # ranking, stability, comparison plots
  orchestration.py       # per-split / fold orchestration, checkpoints
  reports.py             # automated reports, diagnostics, regeneration
  cli.py                 # nested evaluation main()
pipeline/approach2.py    # thin shim → approach2.cli.main
pipeline/approach2_aux.py # thin shim → approach2.extraction
```

### Entry Points (unchanged usage)

```bash
cd pipeline
SANDBOX_API_KEY=... python approach2.py --csv-path ... --out_dir ...
SANDBOX_API_KEY=... python approach2_aux.py --csv-path ... --outdir ... --report-mode mri
```

### Intentionally Deferred (Phase 6)

- Golden-fold regression tests with synthetic extraction CSV fixtures (no API).
- Leakage tests asserting outer-test cases never influence lexicon/calibration training.
- Checkpoint fingerprint compatibility tests for split resume.
- Consolidation of `_coerce_candidate_concepts` between main (`make_slug`) and aux (`.strip()`).

### Validation Performed

```bash
cd pipeline
SANDBOX_API_KEY=dummy python approach2.py --help
SANDBOX_API_KEY=dummy python approach2_aux.py --help
pytest tests/approach2/
```

All approach2 tests passed (including new nested-import smoke tests).

---

## 2026-06-23 — Audit / Current Behavior Review (Prompt 1)

### Scope

Read-only audit of the modular `pipeline/approach2/` package, thin shims (`approach2.py`, `approach2_aux.py`), tests under `pipeline/tests/approach2/`, README, and partial run artifacts in `sabcs/securegpt_dispersion_approach2_pipeline/` (outer splits 001–002 in progress at audit time). No code changes in this pass.

### Files inspected (major)

| Area | Paths |
|------|--------|
| Entry / CLI | `approach2.py`, `approach2_aux.py`, `approach2/cli.py`, `approach2/extraction/cli.py` |
| Splits / orchestration | `approach2/splits.py`, `approach2/orchestration.py` |
| LLM extraction | `approach2/extraction/pipeline.py`, `schema.py`, `data.py`, `config.py`, `prompts/builder.py`, `prompts/extraction.py` |
| API / cost | `approach2/api/client.py`, `approach2/api/cost.py` |
| Features / lexicon | `approach2/lexicon.py`, `recoding.py`, `features/matrices.py`, `features/normalize.py`, `calibration.py`, `audit.py` |
| ML / eval | `approach2/models_ml.py`, `models.py`, `eval_data.py`, `evaluation/plots.py`, `metrics/stats.py` |
| Reports | `approach2/reports.py`, `html_report.py` |
| Tests | `pipeline/tests/approach2/*` (11 files; no leakage or missing-MRI integration tests) |
| Docs | `pipeline/README.md`, this file |
| Run artifacts | `outer_split_*_split_manifest.json`, train extraction checkpoints, partial logs |

### What appears correct

- **Leakage-aware outer-train LLM extraction**: `run_outer_split_for_modality()` calls `extract_subset_records()` on **outer-train row indices only**; held-out cases are featurized via `recode_cases_with_frozen_lexicon()` (regex/phrase matching), not fresh LLM calls.
- **Inner rediscovery on outer-train only**: `build_stable_lexicon_from_training_extractions()` uses `build_rediscovery_subsplits()` over training case IDs only.
- **Frozen artifacts before test scoring**: stable lexicons, recoded matrices, calibration weights, and `GridSearchCV` pipelines (imputer, `LowInfoFeatureFilter`, scaler, model) are fit on outer-train rows; predictions are generated only for outer-test rows in `fit_one_outer_model()`.
- **Split provenance**: each fold writes `{split_id}_split_provenance.csv` and `{split_id}_split_manifest.json` with train/test IDs and hashes (`orchestration.write_split_provenance`).
- **Missing-MRI filtering at ML stage**: `filter_missing_mri_for_dataset()` drops MRI-missing rows for `mri`, `combined`, `mri_pathcal*`, and `mri_teacher_student` before model fitting; pathology-only keeps all cases.
- **Resume layers**: per-case JSON checkpoints under `_case_checkpoints/`; per-split `_split_resume_checkpoint/COMPLETED.json` plus CSV snapshots of predictions/metrics/lexicons.
- **Parallelism**: fold-level (`--parallel-fold-workers`), modality-level, case-level extraction threads, and global API semaphore (`configure_global_api_concurrency`).
- **Cost tracking**: a-priori estimates from real prompts (`estimate_nested_pipeline_llm_cost`); post-run cumulative tracker → `llm_token_cost_report.json`.
- **Reporting**: automated MD/HTML reports, bootstrap CIs, relapse diagnostics, deduplicated aggregate metrics, many plot types under `report_plots/`.

### Bugs, risks, or incomplete behavior

| Issue | Severity | Evidence |
|-------|----------|----------|
| **Default outer scheme is `repeated_mc` (5×80/20)** | Intentional (confirm in docs) | `cli.py` default; live manifest shows `"outer_scheme": "repeated_mc"`. Prompt 2 will document—not change—this default. Use `stratified_kfold` only when explicit disjoint CV is needed. |
| **Repeated-MC dedup averages overlapping test predictions** | Statistical | `deduplicate_outer_predictions()` mean-aggregates when a case appears in multiple outer-test sets; not equivalent to one disjoint CV fold-out per case. |
| **`TEMPERATURE = 0.0` defined but not sent in API payload** | Minor / nondeterminism risk | `extraction/config.py` vs `api/client.py` `_post_chat_completion` (no `temperature` key). |
| **Cost confirmation requires exact `YES` (case-sensitive)** | UX | `confirm_cost_estimate_or_exit()` in `api/cost.py`. |
| **No resume fingerprint for CSV/split args** | Resume safety | `COMPLETED.json` stores metadata but reload does not verify `train_case_hash` against regenerated splits. **Scheduled for Prompt 2.** |
| **No dedicated MRI-missing summary artifact** | Reporting gap | Only log lines `[MISSING_MRI]` and per-case flags in `missed_case_error_analysis.csv`. **Scheduled for Prompt 2:** `mri_missing_case_summary.csv`. |
| **No top-K / cross-modality feature-count balancing** | By design (documented) | Stability threshold only; interpretability report states unequal counts intentional. |
| **Teacher-student pathway: no relapse target** | Incomplete vs spec | `fit_teacher_student_mri_model()` fits dispersion regression + high/low only. **Scheduled for Prompt 2.** |
| **Phase-6 tests deferred** | Test gap | No golden-fold, leakage, or missing-MRI ML-filter tests (noted in progress 2026-06-23). |
| **Test featurization ≠ train featurization** | Methodological | Test uses substring + ontology regex recoding, not LLM. See Prompt 2 plan for permissiveness/miss-risk analysis and coverage diagnostics. |

### Leakage risks (residual)

- Low risk if `--outer-scheme stratified_kfold` and frozen lexicon discipline are used as intended.
- **Repeated MC + mean dedup** can inflate effective sample size and blur held-out interpretation.
- **Fixed ontology patterns** in `SHARED_CONCEPT_ONTOLOGY` apply to test reports (not data-driven; acceptable if treated as prior knowledge).
- **Resume without hash validation** could mix artifacts if `--csv-path`, seed, or split flags change between runs.

### Missing MRI handling (audit)

- Extraction: placeholder record, no API call (`make_missing_extraction_record`).
- Cost estimate: skips missing-text prompts.
- ML: `filter_missing_mri_for_dataset()` after matrix build — **correct for listed pathways**.
- Combined: early-fusion merge then filter — **correct**.
- Gap: no aggregate per-split/per-modality skip counts saved to disk.

### Cost estimation (audit)

- A-priori: builds actual `build_user_prompt()` per scheduled train case; conservative completion cap = `MAX_TOKENS` per call; optional cache-aware prefix model.
- Post-run: parses API `usage` including cached and reasoning tokens; writes `llm_token_cost_report.json`.
- Gaps: no per-split/per-modality cost breakdown in JSON; a-priori uses rough tokenizer heuristic (tiktoken optional).

### Implementation plan (next two prompts)

**Prompt 2 — Correctness, resume safety, teacher–student relapse, and reporting**

**Design defaults (confirm, do not change)**

1. **Keep `repeated_mc` as the default outer scheme** (`--outer-repeats 5`, `--outer-test-frac 0.20`). This matches the intended 80/20 Monte Carlo repeated-split design; do **not** switch the default to `stratified_kfold`. Document in README/methodology that:
   - each repeat is an independent stratified 80/20 draw;
   - `deduplicate_outer_predictions()` mean-aggregates cases that appear in multiple outer-test sets;
   - users who want disjoint one-holdout-per-case CV must explicitly pass `--outer-scheme stratified_kfold --outer-folds 5`.
2. Optionally add a startup log line summarizing outer scheme, repeats, and deduplication rule so runs are self-describing.

**Resume fingerprint validation**

3. On split resume (`load_completed_split_checkpoint`), verify regenerated splits match saved provenance before skipping a fold:
   - recompute `build_outer_splits()` from current `args` + `target_df`;
   - for the requested `split_id`, compare `train_case_hash` / `test_case_hash` (and optionally ordered ID lists) against `{split_id}_split_manifest.json`;
   - also validate `COMPLETED.json` metadata (`csv_path`, `outer_scheme`, `outer_repeats`, `outer_test_frac`, `outer_folds`, `random_seed`) against current CLI args;
   - on mismatch: refuse to load checkpoint, log a clear error, and recompute the split (or exit with instructions to use `--no-skip-completed-splits` after intentional config change).
4. Add tests for fingerprint match/mismatch behavior.

**Teacher–student: relapse prediction**

5. Extend `fit_teacher_student_mri_model()` and `run_one_outer_split()` teacher–student block to produce held-out **relapse** classification predictions (`target_name=relapse_status`, `target_col=relapse_true`), alongside existing dispersion regression and high/low classification.
6. Reuse the same MRI-only feature matrix and pathology teacher signals from outer-train only; do not leak test pathology into student inputs.
7. Add relapse metrics to fold/aggregate outputs and teacher–student sections of automated reports when predictions exist.
8. Skip fold gracefully when relapse train/test class counts are insufficient (same guards as other classifiers).

**Dedicated MRI-missing skip-count summary**

9. Write `mri_missing_case_summary.csv` at run end (and per-split rows where applicable) with at least:
   - overall count of MRI-missing cases in the target-eligible cohort;
   - per `split_id`, per `dataset_key` / pathway: `n_before_filter`, `n_skipped_missing_mri`, `n_after_filter` for train and test partitions;
   - echo totals in `automated_results_report.md/html`.
10. Hook into `filter_missing_mri_for_dataset()` (accumulate counts) or a single post-pass over provenance + raw flags.

**Test featurization via regex / substring recoding — permissiveness and miss risk**

Held-out cases are **not** LLM-extracted. `recode_cases_with_frozen_lexicon()` applies two mechanisms:

| Mechanism | How it matches | Permissiveness |
|-----------|----------------|----------------|
| **Stable phrase features** | Exact normalized substring: `report_norm.find(quote_norm)` after `normalize_text()` (lowercase, collapsed whitespace) | **Strict / conservative.** Not regex-based. Requires verbatim (normalized) quote from train-stable lexicon. |
| **Group / ontology features** | `re.finditer()` over fixed `CANONICAL_GROUP_PATTERNS` (e.g. `\bmultifocal\b`, `\bmeasur`, `\bnme\b`) on normalized report text | **Moderately permissive** within a fixed vocabulary. Some patterns are broad stems; others use word boundaries. Counts all regex hits; negation/uncertainty judged in ±80-char context. |

**Negation / uncertainty context** (both mechanisms): `detect_negation()` / `detect_uncertainty()` on a **±80 character** window using small fixed regex lists (`NEGATION_PATTERNS`, `UNCERTAINTY_PATTERNS` in `config.py`). This is narrower than full-clause clinical negation parsing.

**What can be missed on test (false negatives relative to train LLM extraction)**

- **Unstable phrases**: denovo/seed phrases seen on train but below `--stability-threshold` never enter the frozen lexicon.
- **Paraphrases / synonyms**: test report uses different wording than the frozen `quote_norm` → substring match fails even if clinically equivalent.
- **Normalization drift**: punctuation, line breaks, or spelling differences that survive differently through `normalize_text()` vs LLM quote repair (`schema.py` sanitization on train only).
- **First-match-only phrases**: `.find()` uses the first occurrence; later mentions are ignored for phrase `present`.
- **No novel test-only language**: any finding an LLM would extract on test but that is not in the stable phrase list or ontology regex set is **silently dropped** (features stay zero).
- **Ontology ceiling**: group features cannot fire for concepts outside `SHARED_CONCEPT_ONTOLOGY` patterns unless they were also promoted via stable phrase→group mapping on train.

**Leakage vs coverage tradeoff**

- This design is **intentionally leakage-safe** (test reports are not fed back into discovery) but **not coverage-equivalent** to per-case LLM extraction on test.
- Risk is predominantly **under-featurization / false negatives on held-out cases**, not label leakage. Models may look worse than a full LLM-on-test upper bound would suggest.
- Group ontology recoding can be **broader** than phrase matching (fixed regex list applies to all cases), so group/phrase channels are asymmetric.

**Prompt 2 tasks for test featurization**

11. Add **recode coverage diagnostics** per split/modality:
    - phrase hit rate: fraction of stable phrases with `present>0` on train vs test;
    - fraction of test cases with zero phrase hits but nonzero group hits (and vice versa);
    - optional audit CSV: `recode_coverage_by_split.csv`.
12. Document findings in methodology markdown and interpretability report (do not change recoding logic unless audit shows unacceptable drop-off; if needed later, consider fuzzy quote matching or test-time LLM with frozen lexicon validation as a separate sensitivity analysis).

**Other Prompt 2 items**

13. Fix API payload: pass `temperature` when set; make `YES` confirmation case-insensitive.
14. Add tests: `test_missing_mri_filter.py`, `test_splits_leakage.py` (mirror approach1), `test_resume_fingerprint.py`, golden-fold smoke without API.

**Prompt 3 — Features, reporting & optional enhancements**

1. Configurable stability threshold sensitivity + optional `--top-k-stable-features` per modality (train-only ranking by selection frequency).
2. Spearman scatter plots for top regression models; per-fold performance tables in HTML report.
3. Modality/pathway comparison and calibration-ablation summary plots when ablations enabled.
4. Approach 2 flowchart (mirror Approach 1 `docs/` pattern).
5. Per-split cost accounting in `llm_token_cost_report.json`.
6. Optional sensitivity: test-time LLM extraction with frozen-lexicon-only scoring (only if recode coverage audit warrants it).

---

## 2026-06-23 — Prompt 2: Pipeline correctness, configuration, resume, and parallelism

### Summary

Implemented Prompt 2 core fixes across the modular `pipeline/approach2/` package: case-insensitive cost confirmation, configurable LLM temperature, hardened missing-MRI handling, outer-split validation and provenance reload, permissive configurable stability threshold, optional per-modality stable-feature cap, improved cost estimate/actual accounting, resume fingerprint validation, teacher-student relapse predictions, and parallelism/resume safety improvements.

### Files changed

| File | Changes |
|------|---------|
| `approach2/text_utils.py` | `normalize_yes_no()`, `is_affirmative_response()`, `is_negative_response()` |
| `approach2/api/cost.py` | Case-insensitive confirmation; `write_apriori_cost_estimate_json()`; post-run `cost_type` label; cached-token billing note |
| `approach2/api/client.py` | Pass `temperature` in chat payload; log at init |
| `approach2/extraction/config.py` | `TEMPERATURE` from `LLM_TEMPERATURE` / `TEMPERATURE` env (default 0.0) |
| `approach2/extraction/text_helpers.py` | `_is_missing_text()` treats NA/N/A/nan/none/etc. placeholders as missing |
| `approach2/config.py` | `DEFAULT_STABILITY_THRESHOLD = 0.35` |
| `approach2/eval_data.py` | `has_usable_mri_report()`, `filter_cases_for_modality()`, cohort summary, filter stats tuple, `write_mri_missing_case_summary()` |
| `approach2/splits.py` | `case_id_list_hash()`, `validate_outer_splits()`, `log_outer_split_summary()` |
| `approach2/checkpoint.py` | **New** — split resume fingerprinting, manifest reload, artifact validation |
| `approach2/lexicon.py` | `cap_stable_phrase_lexicon()`, lexicon metadata JSON, `--target-stable-features-per-modality` support |
| `approach2/io_atomic.py` | `atomic_write_json()`, `is_valid_json_file()` |
| `approach2/orchestration.py` | Resume fingerprint + manifest reload; train/test overlap guard; MRI extraction skip; filter stats; teacher-student relapse |
| `approach2/models_ml.py` | Teacher-student optional relapse logistic head on MRI features |
| `approach2/cli.py` | `--temperature`, `--target-stable-features-per-modality`, split validation logging, cohort/MRI summaries, a-priori cost JSON |
| `tests/approach2/test_*.py` | New/updated tests (see Validation) |

### 1. Case-insensitive confirmation

- `confirm_cost_estimate_or_exit()` accepts `yes`/`y` (any case) and rejects `no`/`n`.
- `--yes` flag unchanged for non-interactive runs.

### 2. LLM temperature

- CLI: `--temperature` (default `0.0`); sets `approach2.extraction.config.TEMPERATURE` before API calls.
- API payload key: `temperature` (Azure OpenAI chat-completions).
- Logged at startup, in a-priori estimate JSON, and in split resume fingerprint.

### 3. Missing MRI handling

- Placeholder strings (`NA`, `N/A`, `nan`, `none`, etc.) now count as missing MRI/pathology text.
- MRI extraction skips outer-train cases without usable MRI (no API call).
- `filter_missing_mri_for_dataset()` returns stats; writes `mri_missing_case_summary.csv` and `cohort_report_availability_summary.json`.
- Pathology-only pathways keep all cases; MRI/combined/calibrated/teacher-student drop missing-MRI rows before ML.

### 4. Outer split strategy (unchanged default, now validated)

- **Default remains `repeated_mc`**: 5× stratified 80/20 Monte Carlo draws (`--outer-repeats 5`, `--outer-test-frac 0.20`).
- `validate_outer_splits()` asserts no train/test overlap per split.
- `stratified_kfold`: asserts each case in held-out test exactly once across 5 folds.
- `log_outer_split_summary()` logs per-fold sizes and label balance.
- Resume reloads train/test IDs from `{split_id}_split_manifest.json` when present.
- `deduplicate_outer_predictions()` unchanged (mean-aggregate for repeated MC).

Use disjoint CV explicitly: `--outer-scheme stratified_kfold --outer-folds 5`.

### 5. Stability threshold

- Default `--stability-threshold` lowered from `0.60` → `0.35` (`DEFAULT_STABILITY_THRESHOLD`).
- Threshold logged; per-modality `{split_id}_stable_lexicon_metadata_{mode}.json` saved.

### 6. Normalized stable feature counts

- `--target-stable-features-per-modality N` (default `0` = no cap).
- Train-only ranking by `selection_frequency`, `mean_support_cases`, `n_rows`.
- If fewer stable features than target, uses all available and records note in metadata.

### 7. Cost estimation

- **A-priori**: `llm_cost_estimate_apriori.json` (`cost_type=apriori_estimate`); assumptions documented.
- **Post-run**: `llm_token_cost_report.json` (`cost_type=post_run_actual`); cached tokens not double-counted.
- Temperature included in a-priori estimate payload.

### 8. Resume

- `COMPLETED.json` includes full `fingerprint` (csv, splits, seed, temperature, stability, modalities, etc.).
- Reload validates fingerprint + split manifest hashes before skipping a fold.
- Invalid/empty checkpoint tables trigger recompute.
- Case checkpoints unchanged; atomic writes for marker and CSV snapshots.

### 9. Parallelism

- No behavioral change to fold/modality/API worker model; existing `coordinate_parallelism()` and global API semaphore retained.
- Per-fold isolated directories + resume fingerprint make parallel resume safe.

### 10. Teacher-student relapse

- `fit_teacher_student_mri_model()` adds MRI-only logistic relapse head when train/test class counts sufficient.
- Predictions emitted with `target_name=relapse_status`.

### Bugs fixed

| Bug | Fix |
|-----|-----|
| Cost confirmation required exact `YES` | Case-insensitive yes/y |
| `TEMPERATURE` not sent to API | Added to payload |
| `NA` MRI placeholders treated as usable text | Expanded `_is_missing_text()` |
| Resume ignored config/split changes | Fingerprint + manifest validation |
| No MRI skip summary artifact | `mri_missing_case_summary.csv` |
| Teacher-student missing relapse | Logistic relapse head added |
| No outer split overlap assertion | `validate_outer_splits()` |

### Deferred to Prompt 3

- Recode coverage diagnostics (`recode_coverage_by_split.csv`)
- Per-split cost breakdown in post-run JSON
- Spearman scatter plots, per-fold HTML tables, flowchart
- Golden-fold regression with synthetic extraction fixtures

### Validation performed

```bash
cd pipeline
SANDBOX_API_KEY=dummy .venv/bin/python -m pytest tests/approach2/ -q
# 57 passed

SANDBOX_API_KEY=dummy .venv/bin/python approach2.py --help
SANDBOX_API_KEY=dummy .venv/bin/python approach2_aux.py --help
```

Not run (requires credentials/data): full nested pipeline, live API preflight, resume simulation on partial `sabcs/` artifacts.

### How to validate a real run

```bash
cd pipeline
export SANDBOX_API_KEY=...
python approach2.py \
  --csv-path /path/to/cases.csv \
  --out_dir ./outputs/approach2_prompt2_smoke \
  --modalities mri path \
  --outer-repeats 1 \
  --parallel-fold-workers 1 \
  --yes
```

Check outputs: `mri_missing_case_summary.csv`, `llm_cost_estimate_apriori.json`, `outer_splits/outer_split_001/*_split_manifest.json`, `*_stable_lexicon_metadata_*.json`.

### Suggested next step (Prompt 3)

```bash
# Reporting, recode coverage diagnostics, per-split cost breakdown, plots
cd pipeline && SANDBOX_API_KEY=dummy python approach2.py --regenerate-reports-only --out_dir <existing_out_dir> --csv-path <same_csv>
```

---

## 2026-06-23 — Prompt 3: Reporting, flowchart, README, and HTML review pages (DONE)

### Reporting files added/changed

| File | Change |
|------|--------|
| `approach2/reports.py` | Enhanced results/interpretability reports; added `generate_missed_case_html_report()`, `generate_all_reports()`, Spearman rank plot, pathway comparison, per-fold MAE plot, run-metadata sections |
| `approach2/html_report.py` | Plot captions for new figures; `Dict` import fix |
| `approach2/report_cli.py` | **New** standalone report generator CLI |
| `approach2_generate_reports.py` | **New** thin entry-point shim |
| `approach2/cli.py` | End-of-run reporting delegates to `generate_all_reports()` |
| `scripts/generate_approach2_flowchart.py` | **New** flowchart PNG renderer |
| `docs/approach2_pipeline_flowchart.{mmd,png}` | **New** canonical Approach 2 diagram |
| `README.md` | Approach 2 overview, flowchart, leakage/MRI/split/resume/report commands |
| `tests/approach2/test_report_generation.py` | **New** help, synthetic, and integration tests |

### Flowchart path

`pipeline/docs/approach2_pipeline_flowchart.png` (source: `pipeline/docs/approach2_pipeline_flowchart.mmd`)

### HTML report generator commands

```bash
cd pipeline

# Standalone report generator (recommended)
python approach2_generate_reports.py --run-dir <RUN_DIR> --csv-path <CSV> --force

python -m approach2.report_cli --run-dir <RUN_DIR> --force --open

# Via main pipeline flag (still supported)
python approach2.py --regenerate-reports-only --out_dir <RUN_DIR> --csv-path <CSV>
```

### Generated HTML outputs

| Report | HTML path | Markdown mirror |
|--------|-----------|-----------------|
| Results review | `automated_results_report.html` | `automated_results_report.md` |
| Interpretability review | `interpretability_report.html` | `interpretability_report.md` |
| Missed-case review | `missed_case_review.html` | `missed_case_error_analysis.md` (+ `.csv`) |

### Plots/tables added

- `report_plots/top_regression_spearman_rank.png`
- `report_plots/pathway_modality_comparison.png`
- `report_plots/per_fold_regression_mae.png`
- `interpretability_plots/feature_count_by_modality.png`
- `interpretability_plots/feature_prevalence_by_modality.png`
- `interpretability_plots/top_regression_coefficients.png`
- Results HTML: run overview, split summary, MRI-missing table, label distributions, per-fold metrics table

### Known missing optional artifacts (graceful degradation)

- `mri_missing_case_summary.csv`, `cohort_report_availability_summary.json` — only when full pipeline completes with Prompt 2+ code
- `all_outer_mri_pathology_reliability_matrices.csv`, weighted lexicons — only when `--enable-pathology-calibration`
- Teacher-student / calibration-ablation metrics — only when those flags enabled
- `recode_coverage_by_split.csv`, per-split cost breakdown — still deferred
- Missed-case MRI flags require `--csv-path` for raw report text columns

### Validation performed

```bash
cd pipeline
SANDBOX_API_KEY=dummy .venv/bin/python approach2_generate_reports.py --help
SANDBOX_API_KEY=dummy .venv/bin/python -m pytest tests/approach2/test_report_generation.py -v
SANDBOX_API_KEY=dummy .venv/bin/python approach2_generate_reports.py \
  --run-dir ../sabcs/securegpt_dispersion_approach2 --force
test -f docs/approach2_pipeline_flowchart.png
grep -q approach2_pipeline_flowchart.png README.md
```

Not run: live API pipeline, browser `--open` smoke test.

---

## 2026-06-24 — Multi-model GPT-5 support, reasoning effort, cost reports, validation hardening, temperature removal

### Summary

Today's updates align Approach 2 with Approach 1 on shared LLM infrastructure: selectable `gpt-5-nano` vs full `gpt-5` with model-specific API keys and pricing, GPT-5 `reasoning_effort` in chat payloads, model-aware a-priori and post-run cost JSON, phrase-level quote validation (≤25 words preferred, ≤35 hard cap with repair/drop sanitization), and removal of the deprecated `temperature` parameter.

### 1. Shared multi-model registry (`pipeline/common/`)

Centralized in `common/llm_models.py` and consumed by `approach2/extraction/config.py` via `configure_llm()`:

| Model | Deployment | API key env var | Input / cached / output ($/1M) |
| ----- | ---------- | --------------- | ------------------------------ |
| `gpt-5-nano` (default) | `gpt-5-nano` | `SANDBOX_API_KEY` | $0.05 / $0.01 / $0.40 |
| `gpt-5` | `gpt-5` | `NEW_SECUREGPT_API_KEY` | $1.25 / $0.13 / $10.00 |

CLI flags on both `approach2.py` and `approach2_aux.py`:

```bash
--model {gpt-5-nano,gpt-5}
--env-path /path/to/.env   # must contain the key for the selected model
```

`configure_llm(model, env_path=..., reasoning_effort=...)` sets `MODEL`, `DEPLOYMENT`, `URL`, `HEADERS`, pricing globals, and logs `api_key_env` at init.

### 2. Reasoning effort implementation

Shared normalization in `common/reasoning_effort.py`. Default `medium`; pass `none` to omit from the API request.

| Surface | Behavior |
| ------- | -------- |
| CLI | `--reasoning-effort` on `approach2/cli.py` and `approach2/extraction/cli.py` |
| Env | `REASONING_EFFORT` fallback |
| API | `reasoning_effort` added to payload when non-empty (`approach2/api/client.py`) |
| Resume fingerprint | `reasoning_effort` in `build_split_resume_fingerprint()` (`approach2/checkpoint.py`) |
| Cost JSON | `reasoning_effort` in `llm_token_cost_report.json` and a-priori estimate |

Tests: `tests/common/test_reasoning_effort.py`, `tests/approach2/test_api_client.py`, `tests/approach2/test_resume_fingerprint.py`.

### 3. Cost prediction and post-run report updates

Model-specific pricing now flows through the full cost stack:

- **A-priori** (`estimate_nested_pipeline_llm_cost`, `summarize_apriori_cost_estimate`): prints and writes `llm_cost_estimate_apriori.json` with `model`, `deployment`, `pricing_label`, per-1M rates, `reasoning_effort`, and `cost_type=apriori_estimate`.
- **Post-run** (`write_cost_tracker_json`): `llm_token_cost_report.json` with `cost_type=post_run_actual`, cumulative usage, `billing_note` on cached-token accounting, and completion tokens priced including `reasoning_tokens`.
- **Resume-aware totals**: `initialize_cost_tracker_for_resume()` reloads prior cumulative usage so interrupted runs keep accurate dollar totals.
- **Regression**: `test_estimate_nested_pipeline_llm_cost_uses_selected_model_pricing` asserts GPT-5 no-cache estimate ≈ 25× GPT-5-nano for the same synthetic split plan.

Console a-priori block now prints active `model` and `pricing_per_1M` before the planned-call summary.

### 4. Phrase quote validation (≤25 words target, ≤35 hard cap)

Extraction prompts instruct ≤25 words per quote when possible, never >35 (`approach2/prompts/builder.py`).

Validation pipeline (`approach2/extraction/schema.py`):

1. **Sanitize** (`_sanitize_extraction_obj_for_validation`): per phrase in `seed_aligned_phrases` and `denovo_candidate_phrases`, repair near-miss quotes to exact report substrings or drop the phrase with warnings. Tracks `n_repaired_phrase_quotes`, `n_dropped_phrase_items`, `validation_warnings`.
2. **Validate** (`_validate_phrase_item`): hard cap 35 words; quote must be verbatim in report text under normalized matching.

`approach2/extraction/pipeline.py` runs sanitize → validate → retry; logs `[VALIDATION]` warnings (first 8 per case). Invalid single phrases no longer fail the entire extraction object.

Approach 1 uses the same word-limit policy for `key_evidence` via `approach1/schema/prediction.py` and `approach1/schema/quote_helpers.py`.

### 5. Remove deprecated LLM `temperature` setting

Reverts the 2026-06-23 Prompt 2 temperature wiring. Chat payloads omit `temperature`; sampling follows deployment defaults.

| File | Changes |
|------|---------|
| `approach2/extraction/config.py` | Removed `TEMPERATURE` / `LLM_TEMPERATURE` / `TEMPERATURE` env wiring |
| `approach2/api/client.py` | Removed `temperature` from payload and logs |
| `approach2/cli.py` | Removed `--temperature`, config override, estimate field |
| `approach2/checkpoint.py` | Removed `temperature` from split resume fingerprint |
| `approach2/api/cost.py` | Removed a-priori temperature print |
| `tests/approach2/test_api_client.py` | Assert payload omits `temperature`; asserts `reasoning_effort` when set |
| `tests/approach2/test_resume_fingerprint.py` | Fingerprint uses `model` + `reasoning_effort` instead of `temperature` |
| `README.md` | Removed `TEMPERATURE` env and `--temperature` docs |

**Resume note:** existing `COMPLETED.json` checkpoints that fingerprinted `temperature` may be treated as incompatible on resume.

### Example usage

```bash
cd pipeline
export SANDBOX_API_KEY=...
export NEW_SECUREGPT_API_KEY=...

# Full GPT-5 nested eval
python approach2.py \
  --csv-path /path/to/cases.csv \
  --out_dir ./outputs/approach2_gpt5 \
  --model gpt-5 \
  --reasoning-effort medium \
  --yes

# Standalone extraction with nano
python approach2_aux.py \
  --csv-path /path/to/cases.csv \
  --outdir ./outputs/extract_nano \
  --report-mode mri \
  --model gpt-5-nano \
  --reasoning-effort none \
  -y
```

### Verification

```bash
cd pipeline
grep -ri temperature approach1 approach2 tests/approach1 tests/approach2
SANDBOX_API_KEY=dummy NEW_SECUREGPT_API_KEY=dummy .venv/bin/python -m pytest \
  tests/common tests/approach1 tests/approach2 -q
```
