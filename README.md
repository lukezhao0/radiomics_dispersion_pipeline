# onc-pipeline

Leakage-aware clinical NLP and machine learning pipelines for predicting breast tumor **spatial dispersiveness** and **relapse risk** from post-neoadjuvant, pre-surgery MRI and pathology report text.

Both pipelines use the Stanford Health Care AI Sandbox (GPT-5-nano Global) for LLM-based extraction or prediction, with token/cost tracking, resume/checkpoint support, and PHI-conscious design suitable for clinical research auditing.

## Pipelines at a glance

|                     | **Approach 1**                                                           | **Approach 2**                                                               |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| **Goal**            | Few-shot LLM prediction of dispersion score, high/low class, and relapse | Nested CV: lexical feature discovery → supervised ML → calibration & reports |
| **Entry point**     | `approach1.py`                                                           | `approach2.py`                                                               |
| **Auxiliary entry** | —                                                                        | `approach2_aux.py` (standalone MRI/path extraction)                          |
| **Package**         | `approach1/`                                                             | `approach2/`                                                                 |
| **Change log**      | [`approach1_progress.md`](approach1_progress.md)                         | [`approach2_progress.md`](approach2_progress.md)                             |

**Approach 1** runs a SecureGPT few-shot evaluation across shot sets and modality tiers (MRI, pathology, combined). It predicts continuous dispersion, binary high/low dispersion, and relapse status directly from report text.

### Approach 1 pipeline overview

```mermaid
flowchart TD
    A[Load CSV cases] --> B[Select shot set: 2 high + 2 low exemplars]
    B --> C[Exclude exemplar rows from held-out test]
    C --> D{Modality tier?}
    D -->|mri_only or mri+path| E[Drop MRI-missing test cases]
    D -->|pathology_only| F[Keep all pathology cases]
    E --> G[Build few-shot prompt per test case]
    F --> G
    G --> H[SecureGPT predict JSON]
    H --> I[Validate schema + retry]
    I --> J[Save predictions JSONL/CSV]
    J --> K[Evaluate metrics vs ground truth]
```

**Approach 2** is a nested outer/inner evaluation framework. It extracts quote-grounded lexical features from MRI and pathology reports, discovers stable lexicons within training folds only, trains supervised models for dispersion regression/classification and relapse prediction, and supports pathology-informed MRI calibration, teacher–student pathways, automated reports, and fold-level parallelism.

## Requirements

- Python ≥ 3.10
- Stanford AI Sandbox API key (`SANDBOX_API_KEY`)
- Input CSV with case-level MRI/pathology report text and outcome columns (see each pipeline's expected schema)

## Installation

```bash
cd pipeline
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Set credentials in a `.env` file at the project root (or pass `SANDBOX_ENV_PATH` for approach 2 extraction):

```bash
SANDBOX_API_KEY=your_key_here
```

Optional: `MAX_COMPLETION_TOKENS`, `REASONING_EFFORT`, `TEMPERATURE` (default `0` for deterministic JSON), `SANDBOX_ENV_PATH`.

## Quick start

### Approach 1 — few-shot prediction

```bash
cd pipeline
export SANDBOX_API_KEY=your_key_here

# Interactive (prompts for cost confirmation)
python approach1.py --csv-path /path/to/cases.csv --outdir ./outputs/approach1

# Non-interactive
python approach1.py --csv-path /path/to/cases.csv --outdir ./outputs/approach1 -y
```

Common flags: `--resume` / `--no-resume`, `--skip-completed-configs`, `--force-rerun-cases`, `--skip-preflight`, `--temperature`, `--results-report-only`.

Regenerate the HTML review page without API calls:

```bash
python approach1.py --results-report-only --outdir ./outputs/approach1
```

### Approach 2 — nested lexical + ML evaluation

```bash
cd pipeline
export SANDBOX_API_KEY=your_key_here

python approach2.py \
  --csv-path /path/to/cases.csv \
  --out_dir ./outputs/approach2 \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined \
  --yes
```

Use `--regenerate-reports-only` to rebuild plots and HTML/Markdown reports from saved `nested_outer_*` artifacts without re-running extraction or model fitting.

### Approach 2 — standalone extraction only

Run MRI or pathology lexical extraction without the full nested evaluator:

```bash
python approach2_aux.py \
  --csv-path /path/to/cases.csv \
  --outdir ./outputs/extractions_mri \
  --report-mode mri \
  --max-api-workers 2 \
  --yes
```

## Project layout

```
pipeline/
├── README.md
├── pyproject.toml              # onc-pipeline package (approach1 + approach2)
├── approach1.py                # thin CLI shim → approach1.cli
├── scripts/
│   └── generate_approach1_flowchart.py
├── docs/
│   ├── approach1_pipeline_flowchart.mmd
│   └── approach1_pipeline_flowchart.png
├── approach1/                  # few-shot prediction package
│   ├── cli.py
│   ├── orchestration.py
│   ├── config.py, data.py, splits.py, inference.py
│   ├── api/                    # client, cost tracking
│   ├── prompts/                # system message, templates
│   ├── schema/                 # LLM JSON validation
│   ├── checkpoint/             # resume, fingerprints
│   └── evaluation/             # metrics, plots, evidence
├── approach2.py                # thin CLI shim → approach2.cli
├── approach2_aux.py            # thin CLI shim → approach2.extraction
├── approach2/                  # nested evaluation package
│   ├── cli.py                  # nested evaluation main()
│   ├── orchestration.py        # outer-fold execution, checkpoints
│   ├── reports.py              # automated MD/HTML reports
│   ├── eval_data.py, splits.py, lexicon.py, recoding.py
│   ├── audit.py, calibration.py, models_ml.py
│   ├── config.py, models.py, text_utils.py, io_atomic.py
│   ├── features/               # normalize.py, matrices.py
│   ├── evaluation/             # plots.py, metrics/stats.py
│   ├── extraction/             # LLM extraction (from approach2_aux)
│   │   ├── pipeline.py, schema.py, data.py, cli.py
│   │   └── config.py
│   ├── api/                    # client.py, cost.py
│   ├── prompts/                # extraction templates, builder
│   ├── html_report.py
│   └── logging_setup.py
└── tests/
    ├── approach1/
    └── approach2/
```

Both `approach1.py` and `approach2.py` are **compatibility shims**. New code should import from the `approach1` and `approach2` packages directly.

## Testing

```bash
cd pipeline
pytest tests/approach1/ -v
pytest tests/approach2/ -v

# or all tests
pytest tests/ -v
```

Approach 1 tests cover schema validation, prompt golden strings, split leakage, checkpoint fingerprints, and metric golden values. Approach 2 tests cover config/constants, text utilities, pure metric helpers, import smoke tests, and CLI `--help` (no live API calls).

Verify entry points without a real API key:

```bash
SANDBOX_API_KEY=dummy python approach1.py --help
SANDBOX_API_KEY=dummy python approach2.py --help
SANDBOX_API_KEY=dummy python approach2_aux.py --help
```

## Outputs and logging

- **Approach 1** writes per-config predictions, evaluation plots, `token_cost_report.json`, and a consolidated **`approach1_results_report.html`** under `--outdir`.
- **Approach 2** writes nested CV artifacts under `--out_dir`, including per-split lexicons, predictions, metrics (`nested_outer_metrics_summary.csv`), automated reports (`automated_results_report.md`), interpretability summaries, and logs under `logs/`.
- **Standalone extraction** writes per-case JSON/CSV extractions and `llm_token_cost_report.json`.

See [`approach1_progress.md`](approach1_progress.md) and [`approach2_progress.md`](approach2_progress.md) for detailed output schemas and methodology notes.

## Design principles

- **Leakage awareness** — lexicons, calibration weights, and feature discovery are fit only on outer-training data; held-out test cases receive frozen rules and models.
- **Reproducibility** — explicit seeds, split provenance manifests, checkpoint/resume layers, and versioned progress logs.
- **Modularity** — domain logic lives in installable packages; top-level scripts are thin entry points.
- **PHI consciousness** — quote-grounded extraction, secure API usage, and no real report text in unit tests.

## Further reading

- [`approach1_progress.md`](approach1_progress.md) — API migration, cost estimation, resume/checkpoint behavior, modality tiers.
- [`approach2_progress.md`](approach2_progress.md) — pathology-informed MRI calibration, relapse prediction, fold parallelism, modularization history, and deferred regression-test work.
