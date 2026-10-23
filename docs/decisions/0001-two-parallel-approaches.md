# 0001: Two parallel approaches

Status: Accepted (reconstructed 2026-10-22)

## Context
The question is whether report text can predict tumor dispersion and relapse.
An LLM can be used either to make the prediction directly or to read reports
and propose features for a conventional model. The two have different
strengths: speed and rationales versus transparency and validation rigor.

## Decision
Maintain both:
- **Approach 1** (`approach1/`): few-shot LLM prediction.
- **Approach 2** (`approach2/`): train-only LLM phrase extraction, stable lexicon, regularized ML under nested resampling.

They share `common/` and the same targets and threshold, and are compared
post hoc (`experiment_comparison/`, `metrics_audit/`). The commit history
shows both refactored together (`d0ecd10`).

## Consequences
- Every methodological change has to be considered for both pipelines (threshold, missing-MRI rule, model registry).
- Their metrics are not interchangeable (different cohorts, validation designs and score types). See 0003 and 0011.
- Reason for keeping both **(inferred)**: Approach 1 is the "can the LLM just do it" baseline, and Approach 2 tests whether a transparent model on LLM-proposed features is better or more trustworthy.
