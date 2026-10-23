# 0009: Packages with thin CLI shims

Status: Accepted (commit `d0ecd10` "Pipeline refactoring for approach 1 and approach 2")

## Context
The pipelines began as large single scripts (`feature_discovery_eval_ML*.py`,
`approach1-3.py`, referenced in the progress logs). They were hard to test and
change.

## Decision
- Domain logic lives in installable packages `approach1/`, `approach2/`, `common/` and `experiment_comparison/` (`pyproject.toml`).
- Top-level `approach1.py`, `approach2.py`, `approach2_aux.py` and `approach2_generate_reports.py` only call `main()`.

## Consequences
- Unit tests import package modules directly.
- Many `approach2/*` modules still carry the monolith's large copied import block. Harmless, but noisy.
- Old script names in the progress logs are historical only.
