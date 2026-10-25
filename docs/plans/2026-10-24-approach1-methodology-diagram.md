# Plan: Approach 1 methodology diagram (for non-technical readers)

Date: 2026-10-24. Status: approved and executed the same day. Docs-only; no code touched.

## Goal

Make one simple, friendly diagram (Mermaid) that shows non-technical collaborators
how Approach 1 is built. Raw reports, trimmed reports, run setup, LLM prompt,
answer checking, scoring and the report are drawn as boxes in a chain. Each box
has a plain-language detail box underneath it. The prompt is drawn as a stack of
its sections.

## Files

- Created: `docs/flowcharts/approach1_methodology.md`. It holds the Mermaid diagram, a plain-text version, a glossary, a "what this leaves out" list and a code-location table.
- Created: this plan.
- Not touched: code, prompts, config, tests, `.claude/`, data or output folders, `CLAUDE.md`, `docs/PROGRESS.md`, ADRs. `RESUME_SCRIPT_VERSION` is unaffected because no prompt or schema changed.

## Approach

All facts come from reading the source.

| Step | Fact | Source |
|---|---|---|
| Trimming | Split on blank lines; drop a block whose first line starts with a listed heading | `aux/trim.py:50`, `:110-135`, `:143-160` |
| Input file | A1 default CSV name equals trim.py's output name | `approach1/config.py:12`, `aux/trim.py:43` |
| No further shortening | `shorten_for_prompt` returns text unchanged | `approach1/text_utils.py:49-50` |
| Examples | 4 per shot set (2 high + 2 low), by row number; 2 shot sets | `approach1/config.py:33-44`, `approach1/splits.py:28` |
| Tiers | MRI only, pathology only, both; the missing report shows as `<NOT PROVIDED IN THIS TIER>` | `approach1/config.py:46`, `approach1/prompts/templates.py:33-44` |
| Test set | All rows except the 4 examples; MRI tiers skip cases without MRI | `approach1/splits.py:91-99` |
| Example content | case_id, side, band, true score, high/low, relapse, reports | `approach1/prompts/templates.py:58-67` |
| System message | Sent as the system role | `approach1/prompts/system.py:3-9`, `approach1/api/client.py:63-66` |
| Prompt order | task → predict list → strict rules → descriptors → schema → examples → new case → token | `approach1/prompts/templates.py:92-154` |
| Validation | parse, repair/drop quotes, field/range/label/token-consistency checks, ≤ 3 tries | `approach1/inference.py:26-66`, `approach1/schema/prediction.py:53-188` |
| Evaluation | regression, high/low, relapse, needle, evidence, bootstrap | `approach1/evaluation/runner.py:36-185` |
| Report | aggregate CSV + HTML; rebuildable without API | `approach1/orchestration.py:176-244`, `approach1/cli.py:104-112` |

Corrections to the task's draft:
- There are 4 worked examples, not 2, and they show relapse and high/low labels as well as the score.
- The rules appear in the system message and again after the task.
- The prompt ends with an attention-check token.

## Decisions (from the human, 2026-10-24)

- Location: `docs/flowcharts/`, not `outputs/`. CLAUDE.md treats `outputs/` as PHI-risk.
- Bookkeeping: only the diagram and this plan. No STATUS, log or decisions files; no PROGRESS or CLAUDE.md edits.
- Trimming: drawn as a normal pipeline step, with no caveat in the diagram.
- Format: Mermaid plus a plain-text fallback. No rendered image and no new dependency.
- Git: the agent makes no commits and runs no git commands.

Rejected alternatives:
- Putting the file in `outputs/`, because of the PHI-folder rule.
- Rendering a PNG with mermaid-cli, which needs a new dependency and network access.
- Creating new STATUS, log and decisions files, which would duplicate `docs/PROGRESS.md` and `docs/decisions/`.

## Risks

- PHI: none. The text is derived from code; the descriptor phrases are copied from `descriptors.py`.
- Mermaid layout: nested `subgraph … direction TB` renders differently across viewers. The plain-text version is the fallback.

## Potential problems in the code (logged, not fixed)

1. `case_id` goes into every example, the new-case block and the validation token (`approach1/prompts/templates.py:60`, `:150`; `approach1/prompts/tokens.py:17-22`). If it is an identifier, the API receives it unnecessarily. What `case_id` contains is not verified.
2. The example band comes from row membership, not the score, so the prompt could contradict itself (`templates.py:20-21` vs `:56`). Already noted in PIPELINE_GUIDE §2d.4.
3. Examples are chosen by row position, not by ID (`approach1/splits.py:81`).
4. Relapse AUROC ranks a binary label (`approach1/evaluation/runner.py:53`).
5. Tiers are scored on different cases: pathology-only keeps MRI-missing cases (`approach1/splits.py:96`).
6. Trimming is heading-based and coarse. Nothing in the repo records that it was run (`aux/trim.py`).
7. The validation passes when the model echoes the token wrongly but reports `retrieval_check_correct: false`. Only the self-report's consistency is enforced (`approach1/schema/prediction.py:175-180`). The needle rate records the miss.
8. The hints cover dispersion only. Relapse has no guidance beyond the 4 examples (`approach1/prompts/descriptors.py`).

## Tests and verification

- No code changed, so there are no unit tests. `pytest tests/ -q` is unaffected and was not run.
- Human check: open `docs/flowcharts/approach1_methodology.md` in GitHub or the VS Code Markdown preview (with Mermaid support). Pass: 7 columns left to right, each with a detail box under its step box, and the prompt stack as 7 boxes joined by arrows. Fail: a parse error, or the boxes laid out sideways.
