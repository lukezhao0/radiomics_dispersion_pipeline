# Working in `onc-pipeline`

The Data Safety Rules in the root `CLAUDE.md` take priority over everything here.

## Purpose

Clinical NLP / ML research code that predicts breast-tumor **spatial
dispersiveness** (a continuous score, also dichotomized at 85) and **relapse**
from post-neoadjuvant, pre-surgery MRI and pathology report text. Two
approaches are compared. **Approach 1** asks an LLM for direct few-shot
predictions. **Approach 2** uses an LLM to discover quote-grounded phrases,
then fits classical regularized models under nested resampling. Both call the
Stanford Health Care AI Sandbox (gpt-5-nano or gpt-5).

## Setup (human)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # deps in pyproject.toml; Python >= 3.10
```

API keys are read from a `.env` file (default set by `ENV_PATH` in the
`config.py` modules and `common/llm_models.py`, outside this repo; override
with `--env-path`, `SANDBOX_ENV_PATH` or `ENV_PATH`):
`SANDBOX_API_KEY` (gpt-5-nano) and `NEW_SECUREGPT_API_KEY` (gpt-5). Never read
or print `.env`.

## Directory map

| Path | What it is |
|------|------------|
| `approach1.py` | CLI shim → `approach1.cli:main` |
| `approach1/` | Few-shot LLM prediction: `config.py` (shot sets, threshold, tiers), `data.py`, `splits.py`, `prompts/`, `inference.py`, `schema/` (JSON validation), `checkpoint/` (resume), `evaluation/` (metrics, bootstrap, plots, HTML), `orchestration.py` |
| `approach2.py` | CLI shim → `approach2.cli:main` (nested evaluation) |
| `approach2_aux.py` | CLI shim → `approach2.extraction` (extraction only) |
| `approach2_generate_reports.py` | CLI shim → `approach2.report_cli` (reports only) |
| `approach2/` | Nested lexical discovery + ML: `eval_data.py`, `splits.py`, `extraction/`, `prompts/`, `features/normalize.py`, `lexicon.py`, `recoding.py`, `features/matrices.py`, `calibration.py`, `models.py`, `models_ml.py`, `orchestration.py`, `reports.py`, `checkpoint.py`, `config.py` (ontology, threshold, seed) |
| `common/` | Shared: `llm_models.py` (model registry, pricing, keys), `reasoning_effort.py`, `bootstrap_cis.py`, `cost_comparison.py` |
| `experiment_comparison/` | Compare completed run dirs from a YAML config (no re-fit). See its README |
| `metrics_audit/` | Offline combined-modality (MRI+pathology) A1 vs A2 audit. See its README |
| `scripts/` | Flowchart renderers, offline A1 bootstrap, unified combined-modality eval |
| `tests/` | pytest, synthetic fixtures only: `approach1/`, `approach2/`, `common/` |
| `docs/` | This file, architecture, guide, progress, glossary, ADRs, flowcharts |
| `approach1_progress.md`, `approach2_progress.md` | Historical change logs (dated entries; code supersedes them) |

## How to run each stage (FOR THE HUMAN TO RUN; the agent must not run these)

Substitute your own CSV and output paths. Live runs call the paid API and
write PHI-bearing outputs to the output directory.

```bash
# Approach 1: few-shot prediction (all shot sets x 3 tiers)
python approach1.py --csv-path <CSV> --outdir <OUT_A1> --model gpt-5-nano --reasoning-effort medium -y
# Approach 1: rebuild metrics, bootstrap CIs, HTML from saved predictions (no API)
python approach1.py --results-report-only --outdir <OUT_A1>

# Approach 2: nested evaluation
python approach2.py --csv-path <CSV> --out_dir <OUT_A2> --modalities mri path combined --yes
# Approach 2: rebuild reports from saved artifacts (no API)
python approach2_generate_reports.py --run-dir <OUT_A2> --csv-path <CSV> --force
# Approach 2: extraction only
python approach2_aux.py --csv-path <CSV> --outdir <OUT_X> --report-mode mri --yes

# Post-hoc (no API)
python -m experiment_comparison.run_comparison --config experiment_comparison/configs/<cfg>.yaml
python metrics_audit/compute_unified_metrics.py --approach1-dir <OUT_A1> --approach2-dir <OUT_A2> --outdir <OUT_AUDIT>

# Tests and help (synthetic data, no API)
pytest tests/ -q
SANDBOX_API_KEY=dummy python approach2.py --help
```

## Conventions seen in the code

- Domain logic lives in packages. Top-level `*.py` files are thin shims.
- `from __future__ import annotations` plus type hints. Module docstring on line 1.
- Console logging via `print` with bracketed tags (`[RUN]`, `[RESUME]`, `[OUTER]`, `[SKIP]`, `[WARN]`). stdout/stderr are tee'd to `run.log`.
- Atomic writes for checkpoints (`io_atomic.py`). Resume is guarded by fingerprints (`RESUME_SCRIPT_VERSION` in `approach1/config.py` and `approach2/checkpoint.py`).
- Constants in `config.py` modules. Argparse defaults in `cli.py`. These can drift apart; see the stability-threshold note in `docs/PROGRESS.md`.
- Many `approach2/*` modules start with the same large copied import block, left over from the monolith split.
- Tests use synthetic `SYNTH_*` cases and never call the live API.

## Traps to know before editing

- **Stability threshold:** the `--stability-threshold` argparse default is 0.60 (`approach2/cli.py`). The constant `DEFAULT_STABILITY_THRESHOLD` and `docs/approach2_feature_stability_and_selection.md` say 0.20.
- **Metric semantics:** Approach 1 high/low AUROC ranks `dispersion_score_pred`. Approach 1 relapse AUROC ranks a **binary** 0/1 label. Approach 2 AUROC ranks model probabilities. These are not the same quantity.
- **Prompt or schema edits** change scientific outputs and can invalidate checkpoints. Say whether `RESUME_SCRIPT_VERSION` must change.
- **Leakage boundary:** in Approach 2, nothing learned from data (lexicon, weights, scaler, model) may see outer-test cases.
- **Missing MRI** is an explicit exclusion, never a zero vector.

## Other docs

- `docs/ARCHITECTURE.md`: components, data flow, how stages connect
- `docs/PIPELINE_GUIDE.md`: beginner walkthrough with diagrams
- `docs/PROGRESS.md`: current state, open issues, next steps, proposed layout changes
- `docs/GLOSSARY.md`: domain and statistical terms
- `docs/decisions/`: ADRs for the main design choices
- `docs/approach2_feature_stability_and_selection.md`: method note on rediscovery and feature counts
- `experiment_comparison/README.md`, `metrics_audit/README.md`

## How to work here

1. **Plan first.** Read the relevant code and state the plan (files to touch, risks: leakage, checkpoint invalidation, metric-definition change, PHI, API cost) before editing. Wait for approval on anything non-trivial.
2. **Small changes.** One logical change at a time. No drive-by refactors or renames.
3. **Commit after each step** with a clear message, on a feature branch. Never push.
4. **Verification is the human's job** whenever it touches data or the API: write the exact command and expected pass/fail signal.
5. **End of session:** update `docs/PROGRESS.md` (dated entry: what changed, what's next, open questions). Record any new design choice as an ADR in `docs/decisions/`.
