# Experiment Comparison Framework

Config-driven tooling to compare completed Approach 1 and Approach 2 pipeline runs across performance metrics, LLM cost, token usage, and experimental settings (shotset, modality, reasoning effort, model family, feature representation, pipeline version).

Designed for two goals:

1. **Immediate use** — compare specific completed runs (e.g. SABCS 0623–0624 experiments) in one HTML report.
2. **Future reuse** — add new run directories or manual legacy metrics via YAML/JSON without changing Python code.

## How it works

The comparison pipeline is fully config-driven:

```
YAML/JSON config
    → discover artifacts in each run directory
    → extract & normalize metrics (long-form table)
    → compute per-run “best” metrics (documented rules)
    → generate comparison plots
    → write HTML report + intermediate CSVs
```

### Core modules

| Module | Role |
|--------|------|
| `load_config.py` | Load config, validate fields, resolve paths relative to config file |
| `discover_results.py` | Walk run dirs; catalog CSV, JSON, HTML, log artifacts |
| `extract_metrics.py` | Parse known result tables; normalize metric names; preserve provenance |
| `normalize_results.py` | Long-form dataframe; best-metric selection; availability summary |
| `plot_comparisons.py` | 13 comparison plots (PNG); graceful handling of missing data |
| `generate_report.py` | Standalone HTML report with tables, plots, captions, limitations |
| `run_comparison.py` | CLI entry point |

### What gets extracted

**Approach 1** (from each run directory):

- `all_tiers_metrics_summary.csv` — per shotset × modality metrics (including `dispersion_high_low_auroc` / `dispersion_high_low_auprc`)
- `evaluation_metrics_summary.json` — detailed eval per config (`dispersion_high_low`, relapse predictor comparison, etc.)
- `llm_token_cost_report.json` and/or per-config `token_cost_report.json` (summed when no run-level aggregate exists)
- `run.log` — runtime estimate when parseable

**Approach 1 high/low metric semantics** (important for cross-run plots):

| Metric | Source | Meaning |
|--------|--------|---------|
| `accuracy`, `f1` | `dispersion_high_low_pred` | LLM's explicit binary high/low label |
| `auroc`, `auprc` | `dispersion_score_pred` | Ranking quality of continuous predicted score vs true high/low |

Approach 2 high/low AUROC uses classifier probabilities from `nested_outer_metrics_summary.csv`. The comparison report plots **both** AUROC and accuracy side-by-side so these distinct quantities are not mixed on one axis.

**Approach 2**:

- `nested_outer_metrics_summary.csv` — regression/classification metrics with bootstrap CIs, by `dataset_key`, `representation`, `model_key`, `target_name`
- `llm_token_cost_report.json`, `llm_cost_estimate_apriori_initial.json` (preferred), `llm_cost_estimate_apriori.json` (session-only fallback)

**Manual legacy results** (no directory required):

- Entries under `manual_results` in the config with user-supplied metric values (e.g. old-pipeline GPT-5 benchmarks)

Every extracted value retains `source_file`, `raw_metric`, and optional CI bounds. Conflicting values from different files are kept, not silently overwritten.

### Best-metric rules

Documented in `constants.py` and reflected in `summary_best_metrics.csv`:

- **Higher is better:** Spearman ρ, AUROC, AUPRC, accuracy, F1, precision, recall, specificity, NPV, R², Pearson r
- **Lower is better:** MAE, RMSE, Brier, cost, runtime

Within each run, “best” is the max/min across shotsets, modalities, representations, and model families unless a plot stratifies by those dimensions.

### Plots and scientific questions

The report includes plots (when data is available) for:

1. Best regression Spearman ρ by run
2. Best high/low classification — **dual panel**: AUROC (left) and accuracy (right); Approach 1 AUROC uses `dispersion_score_pred`, Approach 2 uses model probabilities
3. Best relapse classification
4. Approach 1 shotset comparison
5. Modality comparison (MRI-only, pathology-only, combined)
6. Reasoning-level comparison (minimal / low / medium)
7. Approach 2 model-family comparison
8. Approach 2 feature-representation comparison
9. Cost comparison (a priori vs actual)
10. Performance vs cost
11. Performance vs runtime / API calls
12. API-call and token-usage comparison
13. Summary heatmap (normalized key metrics)

Captions in the HTML report tie each plot to the scientific question it addresses.

## Quick start

From the `pipeline/` directory:

```bash
source .venv/bin/activate   # if not already active
pip install -e ".[dev]"     # includes pyyaml

python -m experiment_comparison.run_comparison \
  --config experiment_comparison/configs/comparison_0623_0624.yaml
```

Outputs are written to the path in `output.output_dir` (relative to the config file). For the bundled example:

```
experiment_comparison/output/comparison_0623_0624/
├── comparison_report.html          # main deliverable
├── normalized_metrics_long.csv
├── summary_best_metrics.csv
├── data_availability_summary.csv
├── discovered_files.csv
├── raw_extracted_metrics.csv
└── plots/*.png
```

### Backfilling Approach 1 metrics after pipeline updates

If an Approach 1 run directory predates high/low AUROC support, refresh metrics from saved predictions (no API calls):

```bash
python approach1.py --results-report-only --outdir /path/to/approach1_run_dir
```

This re-runs `evaluate_and_plot` on each config's `predictions_testing_cases.csv`, updates `evaluation_metrics_summary.json` and `all_tiers_metrics_summary.csv`, then rebuilds `approach1_results_report.html`. Re-run the comparison CLI afterward.

## Configuration

Example: `configs/comparison_0623_0624.yaml`

```yaml
runs:
  - id: approach1_062426_nano_low
    path: ../../../sabcs/securegpt_dispersion_approach1_pipeline_062426
    label: "Approach 1 062426 Nano Low"
    approach: "approach1"
    model: "gpt-5-nano"
    reasoning: "low"
    pipeline_version: "new_refactored"
    notes: "Optional free-text note"

manual_results:
  - id: approach1_old_gpt5_medium
    label: "Approach 1 Old GPT5 Medium"
    approach: "approach1"
    model: "gpt-5"
    reasoning: "medium/default"
    pipeline_version: "old"
    metrics:
      high_low_accuracy_best: 0.8061
      regression_spearman_best: 0.7798

output:
  output_dir: ../output/comparison_0623_0624
  html_filename: comparison_report.html

plots:
  - best_regression_spearman
  - best_high_low_classification
  # ... see constants.DEFAULT_PLOTS for full list

metric_aliases: {}   # optional overrides for canonical metric names
```

### Adding a new run

1. Copy or edit a config YAML.
2. Add a `runs` entry with `path` relative to the config file.
3. Set annotations: `approach`, `model`, `reasoning`, `pipeline_version`, `label`.
4. Re-run the CLI.

### Adding manual / legacy metrics

Add a `manual_results` entry with a `metrics` dict. Supported keys include:

- `regression_spearman_best`
- `high_low_accuracy_best`
- `high_low_auroc_best`
- `relapse_auroc_best`

Custom keys are stored but may not appear in all plots.

## Current comparison (0623–0624)

The example config compares five directory runs under `sabcs/`:

| Run | Approach | Model | Reasoning | Pipeline |
|-----|----------|-------|-----------|----------|
| `securegpt_dispersion_approach1_pipeline_062326` | 1 | GPT-5-nano | medium/default | new refactored |
| `securegpt_dispersion_approach2_pipeline_062326` | 2 | GPT-5-nano | minimal | new refactored |
| `securegpt_dispersion_approach1_pipeline_062426` | 1 | GPT-5-nano | low | new refactored |
| `securegpt_dispersion_approach2_pipeline_062426` | 2 | GPT-5-nano | low | new refactored |
| `securegpt_dispersion_approach2_pipeline_062426_medium` | 2 | GPT-5-nano | medium/default | new refactored |

Plus two manually entered legacy GPT-5 (old pipeline) reference points.

## Known limitations

The framework is robust to missing files and metrics; gaps are listed in the HTML report **Limitations** section. Common gaps:

- **Approach 1 high/low AUROC on old run dirs** — requires a one-time `--results-report-only` backfill (see above) before comparison will include Approach 1 on the AUROC panel.
- **Legacy manual Approach 1 metrics** — the bundled config supplies accuracy only for old GPT-5; AUROC appears on the accuracy panel unless you add `high_low_auroc_best` to `manual_results`.
- **Wall-clock runtime** — only estimated when run logs contain parseable session timestamps.
- **A priori cost** — prefers `llm_cost_estimate_apriori_initial.json` (immutable full-pipeline plan). Falls back to `llm_cost_estimate_apriori.json` only when that snapshot is a fresh-run estimate with no resume skips.
- **Teacher–student / ablation pathways** — compared only if present as `dataset_key` values in `nested_outer_metrics_summary.csv`.

## Design principles

- **No hardcoded run paths** in Python (only in example configs).
- **Provenance preserved** for every metric (`source_file`, `raw_metric`, extraction confidence).
- **No silent deduplication** of conflicting values unless a documented aggregation rule applies (e.g. summing per-config Approach 1 costs).
- **No hallucinated metrics** — missing data is warned, not invented.

## Suggested follow-ups

- Parse `approach1_results_report.html` / `automated_results_report.html` for additional embedded metrics.
- Add explicit runtime fields to pipeline outputs for reliable performance-vs-runtime plots.
- Unit tests with small fixture CSV/JSON snippets per approach.
- Optional SVG export and old-vs-new pipeline version faceting in plots.
- Add `high_low_auroc_best` to manual legacy entries where historical AUROC was computed offline.
