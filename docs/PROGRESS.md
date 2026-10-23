# Progress

Living status document. Update it at the end of every session with a dated
entry in the [Session log](#session-log). Detailed implementation history
stays in `approach1_progress.md` and `approach2_progress.md`. Where they
disagree, the code wins.

**Doc conventions.** `docs/AGENT_PROMPT_TEMPLATE.md` (untracked as of
2026-10-22) describes a different scheme. This repo uses the following mapping:

| Template name | Used here |
|---|---|
| `docs/STATUS.md` | this file, "Current state" |
| `docs/log.md` | this file, "Session log" |
| `docs/decisions.md` | `docs/decisions/NNNN-*.md` (ADRs) |
| `docs/plans/` | not created yet; add plans there if you adopt the template |

---

## Current state (as of 2026-10-22)

- **Last code commit:** `427e1fd` (2026-06-29). The work has been idle for about 4 months.
- **Approach 1:** functional end to end.
  - 2 shot sets × 3 tiers; gpt-5-nano or gpt-5; reasoning effort; resume.
  - Score-based high/low AUROC/AUPRC (`1bd4416`) and case-level bootstrap CIs (`58c92a4`).
  - `--results-report-only` backfills metrics for old run dirs.
- **Approach 2:** functional end to end.
  - Nested repeated-MC evaluation; pathology calibration and teacher-student (both on by default); fold, modality and API parallelism; resume.
  - 3 HTML reports.
  - Commit `9a759ce` (GPT-5 and minimal reasoning for Approach 2) is marked "NOT FULLY TESTED". Later commits say "working", but no explicit retest is recorded.
- **Post-hoc tools:**
  - `experiment_comparison/`: working (`34e2190`), one bundled config (`comparison_0623_0624.yaml`).
  - `metrics_audit/`: combined-modality A1 vs A2 audit with pre-specified headline models (`5aa4d68`, ADR 0011). Its README headline numbers are from one June 2026 run and are not a standing result.
- **Tests:** synthetic pytest suites in `tests/approach1`, `tests/approach2`, `tests/common`. Their pass/fail status was not checked in this session (agent does not run code).
- **No TODO/FIXME comments** in the code. Unfinished work is recorded in the progress logs and READMEs instead.

## Unfinished / deferred work

From `approach2_progress.md` ("Deferred to Prompt 3", partially done) and the
`experiment_comparison/README.md` "Suggested follow-ups":

- [ ] Recode coverage diagnostics (`recode_coverage_by_split.csv`): how many test cases get zero features. No code for it found.
- [ ] Per-split cost breakdown in the post-run cost JSON.
- [ ] Golden-fold regression test with synthetic extraction fixtures.
- [ ] `experiment_comparison` unit tests with fixture CSV/JSON; explicit runtime fields in pipeline outputs; legacy `high_low_auroc_best` entries.

## Open issues found in the 2026-10-22 code read

Severity is my judgment. Verify before acting; nothing here was executed.

| # | Issue | Where | Severity |
|---|-------|-------|----------|
| 1 | `--stability-threshold` default is 0.60, but the constant and the method doc say 0.20 (ADR 0010) | `approach2/cli.py:363`, `approach2/config.py` | High: ambiguous what past runs used |
| 2 | With default `stable_plus_ontology`, every ontology group is recoded regardless of stability, so `group_*` features barely depend on LLM extraction (inferred) | `approach2/orchestration.py` (`extra_groups`), `recoding.py` | High: affects interpretation of A2 |
| 3 | Repeated MC 5×20% leaves about a third of cases without any held-out prediction; pooled metrics cover a subset | `approach2/splits.py`, `orchestration.py:deduplicate_outer_predictions` | High for A1-vs-A2 comparison |
| 4 | Reports pick the best of many dataset×representation×model rows on the same test predictions (optimistic) | `orchestration.py:_best_row`, `reports.py`, `experiment_comparison` | Medium: label as exploratory |
| 5 | Teacher-student `y_prob = clip((pred-85+20)/40)` is not a probability and creates ties; Brier/calibration are meaningless for it | `approach2/models_ml.py:fit_teacher_student_mri_model` | Medium |
| 6 | Teacher-student exceptions are swallowed with a `[WARN]` print | `approach2/orchestration.py` | Medium |
| 7 | `--weighted-lexicon-min-selection-frequency` is a no-op (`stab = max(stab, min)` then `if stab < min`) | `approach2/calibration.py:compute_weighted_mri_lexicon` | Low (default 0.0) |
| 8 | A1 sends unlabeled-dispersion cases to the API; A2 drops them first, so the cohorts differ | `approach1/splits.py`, `approach2/eval_data.py` | Medium |
| 9 | A1 exemplar band label comes from row lists, not the score; it could contradict the score line (unverified) | `approach1/prompts/templates.py:exemplar_dispersion_band` | Medium if it occurs |
| 10 | A1 exemplars are chosen by `iloc` row number; a re-sorted CSV silently changes them, and the fingerprint wouldn't notice | `approach1/config.py:SHOT_SETS`, `checkpoint/fingerprint.py` | Medium |
| 11 | A1 resume fingerprint ignores prompt/schema text; stale predictions are reused unless `RESUME_SCRIPT_VERSION` is bumped | `approach1/checkpoint/fingerprint.py` | Medium |
| 12 | A2 recoding ignores `index_side`; contralateral findings can become features | `approach2/recoding.py` | Medium |
| 13 | Negation/uncertainty via ±80-character regex window; phrase check only at first occurrence | `approach2/recoding.py` | Low–Medium |
| 14 | Extracted `quantitative_attributes`, `report_level_summary`, `laterality` and `directness` are never used as features | `approach2/features/` | Low (cost / missed signal) |
| 15 | Outer splits stratified on dispersion only; relapse balance is unenforced | `approach2/splits.py` | Low–Medium |
| 16 | `SVC(probability=True)` has no `random_state`, so its Platt scaling is not reproducible | `approach2/models_ml.py` | Low |
| 17 | Bootstrap CIs ignore re-splitting, re-extraction and model-selection variability | both | Low (document it) |
| 18 | Default CSV path in A1 config (`…/onc/PROCESSED_…csv`) differs from the docstring/README path (`…/onc/data/…`) | `approach1/config.py`, `approach2/extraction/config.py`, `approach1.py` | Low |
| 19 | A1 default `OUT_DIR` is `./securegpt_dispersion_3tier_outputs` (cwd-relative, not gitignored until this branch) | `approach1/config.py` | Medium (PHI could land in repo) |
| 20 | `run.log` captures quote snippets (≤160 chars) and raw model output (first 400 chars) on validation failures | `approach1/inference.py`, `schema/prediction.py` | Medium (PHI in logs) |
| 21 | `metrics_audit/compute_unified_metrics.py` and `scripts/unified_combined_modality_eval.py` overlap (same commit) | both | Low |
| 22 | The A2 extraction prompt says "≤ 30 words" while validation targets 25 with a cap of 35 | `approach2/prompts/builder.py` | Low |

## Suggested next steps (in order)

1. **Pin down what past runs used:** stability threshold (issue 1), model, reasoning effort, outer scheme. Read each run's lexicon metadata JSON and `run.log` header.
2. **Decide on group features (issue 2):** is "ontology-driven" intended? If not, rerun with `--ontology-groups-mode stable_only` as a sensitivity analysis.
3. **Make A1 and A2 comparable:**
   - A2 with `--outer-scheme stratified_kfold` for full coverage.
   - The same label filter in both.
   - Matched-case metrics.
4. **Fix the small correctness bugs** (issues 5, 6, 7, 16), each with a failing synthetic test first.
5. **Reduce PHI exposure in outputs/logs** (issues 19, 20): default outputs under a gitignored folder; truncate or omit quote text in console warnings.
6. **Add the deferred golden-fold test and recode-coverage diagnostic.**
7. **Repo hygiene and layout** (below).

## Open questions for the project owner

- What is the dispersion score (units, how it was measured), and why the 85 cutoff? (ADR 0005)
- Can a patient contribute more than one row (e.g. bilateral disease)? If so, splits should group by patient.
- Which stability threshold did the June runs use, and which should be canonical?
- Are `group_*` features *meant* to be ontology-only in the default mode?
- Is relapse a primary or secondary endpoint? This affects whether splits should stratify on it.

---

## Repo hygiene findings (2026-10-22)

Found with `git ls-files` (paths and extensions only; no contents read).

**Tracked data-type files**
- `experiment_comparison/output/comparison_0623_0624/`: 5 `.csv`, 1 `.html`, 13 `plots/*.png`.
- `experiment_comparison/configs/output/comparison_0623_0624/`: the same set again. Apparently an older output-path resolution; the bundled config now writes to `experiment_comparison/output/`.
- `.mplconfig/`: 1 `.json` (matplotlib cache).
- `onc_pipeline.egg-info/`: 4 `.txt` + `PKG-INFO` (build metadata).
- These are all pushed to `origin/main`. Whether the comparison CSV/HTML hold only aggregate metrics has **not** been verified.

**`.gitignore`**
- This branch adds data extensions, data/output/log folder names, the A1 default output dir, notebook checkpoints, build and cache dirs, and `.env*`.
- Ignoring does not untrack existing files. To stop tracking them, the human would run something like:

```bash
git rm -r --cached experiment_comparison/output experiment_comparison/configs/output .mplconfig onc_pipeline.egg-info
```

- If the comparison outputs contain anything sensitive, removal from history (e.g. `git filter-repo`) and a force push would also be needed. That is the human's decision.

**Suggested tooling (not installed)**
- `nbstripout`: strips notebook outputs on commit (`pip install nbstripout && nbstripout --install`). There are no notebooks today; this is preventive.
- A `pre-commit` config blocking data files. Example `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: detect-private-key
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: local
    hooks:
      - id: block-data-files
        name: Block data/PHI-prone files
        entry: "Data-type files must not be committed"
        language: fail
        files: '(\.(csv|tsv|xlsx?|parquet|feather|pkl|pickle|rds|RData|h5|hdf5|hl7|dcm|nii|log)$)|(^|/)(data|raw|outputs?|results|logs|cache|tmp|exports)/'
```

---

## Proposed layout changes

**Proposals only.** Nothing has been moved, renamed or deleted. Each lists
every reference that would need updating (found by `git grep` on 2026-10-22).

1. **Untrack generated comparison outputs** (`experiment_comparison/output/`, `experiment_comparison/configs/output/`).
   - Why: generated artifacts, possible PHI risk, and duplicated.
   - References: `experiment_comparison/configs/comparison_0623_0624.yaml:74` (`output_dir: ../output/...`, can stay), `experiment_comparison/load_config.py:139` (default `"output"`, can stay), `experiment_comparison/README.md:105,155`, and the README layout comment "generated reports (gitignored in practice)" (would become true).
   - Code change: none.
2. **Untrack `onc_pipeline.egg-info/` and `.mplconfig/`.**
   - Why: build and cache artifacts.
   - References: none in code. The `README.md` metrics-audit example sets `MPLCONFIGDIR=/tmp/mplcache`, so `.mplconfig/` may come from a local env var; check before ignoring.
3. **Consolidate `scripts/unified_combined_modality_eval.py` into `metrics_audit/`.**
   - Why: two overlapping implementations of the combined-modality audit, added in the same commit (`5aa4d68`).
   - References: no other file references the script; it imports from `approach1`, `approach2` and `common`. `metrics_audit/README.md` documents only `compute_unified_metrics.py`.
   - Decide which one is canonical first.
4. **Move `approach1_progress.md` / `approach2_progress.md` to `docs/history/`.**
   - Why: root clutter; they are historical logs.
   - References: `README.md:15,373,384,385`; `docs/approach2_feature_stability_and_selection.md:349`; `docs/CLAUDE.md` (directory map); `docs/PROGRESS.md` (this file); `docs/PIPELINE_GUIDE.md` (mentions); `docs/AGENT_PROMPT_TEMPLATE.md` (untracked); `onc_pipeline.egg-info/PKG-INFO` (regenerated).
5. **Move flowcharts to `docs/flowcharts/`.**
   - Why: separate generated images from prose docs.
   - References: `scripts/generate_approach1_flowchart.py:4,24,25,27`; `scripts/generate_approach2_flowchart.py:4,22,23`; **`tests/approach2/test_report_generation.py:233`** (the test would fail if not updated); `README.md:289-292`; `approach1_progress.md:294,296,404`; `approach2_progress.md:1086,1092,1142,1143`; `docs/approach2_feature_stability_and_selection.md:350`.
   - Lower value than items 1–4.
6. **Default output location inside a gitignored `outputs/`.**
   - Why: the A1 default (`./securegpt_dispersion_3tier_outputs`) lands in the repo root when run from there.
   - This is a code change (`approach1/config.py:OUT_DIR`), not a move. The `.gitignore` on this branch already covers both names.

---

## Session log

### 2026-10-22: Claude Code onboarding (branch `setup/claude-onboarding`)

- Read-only code review; no code executed, no data files opened.
- Added root `CLAUDE.md` (data safety rules plus known PHI-risk paths) and `docs/CLAUDE.md`, `ARCHITECTURE.md`, `PIPELINE_GUIDE.md`, `GLOSSARY.md`, `decisions/0001–0011`, and this file. Extended `.gitignore`.
- A `PreToolUse` PHI hook initially blocked the root `CLAUDE.md`. The owner adjusted the hook, and data-safety rule 1 now says only "do not access files outside this working directory", without naming external locations.
- Not done: running tests, verifying the issues above, untracking files, any code changes.
- Next session should read: this file, then `docs/decisions/0010` and issues 1–3.

**Verification commands for the human** (no PHI printed):

```bash
# tests and CLI help (synthetic only)
.venv/bin/pytest tests/ -q
SANDBOX_API_KEY=dummy .venv/bin/python approach1.py --help >/dev/null && echo ok

# issue 9: do exemplar band labels match the score? prints booleans only
.venv/bin/python -c "import pandas as pd;from approach1.config import SHOT_SETS as S;d=pd.read_csv('<CSV>');c='dispersion_invasive_DCIS_geographic';[print(s['name'],[bool(d.iloc[r][c]>=85) for r in s['high_rows']],[bool(d.iloc[r][c]<85) for r in s['low_rows']]) for s in S]"

# issue 1: which stability threshold did a run use? prints numbers only
grep -ho '"stability_threshold": [0-9.]*' <OUT_A2>/outer_splits/*/*/*_stable_lexicon_metadata_*.json | sort | uniq -c

# possible duplicate patients (one row per patient?) prints counts only
.venv/bin/python -c "import pandas as pd;d=pd.read_csv('<CSV>');print(len(d), d['case_id'].nunique() if 'case_id' in d else 'no case_id')"

# tracked comparison outputs: inspect headers yourself for aggregate-only content
git ls-files 'experiment_comparison/*output*/*.csv'
```
