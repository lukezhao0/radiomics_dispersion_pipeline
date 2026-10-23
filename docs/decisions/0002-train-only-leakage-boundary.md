# 0002: Train-only leakage boundary in Approach 2

Status: Accepted (documented in README "Design principles" and `approach2_progress.md`)

## Context
Approach 2 learns several things from data: which phrases matter (LLM
extraction + rediscovery), MRI–pathology reliability weights, a pathology
teacher, scalers and model weights. If any of these see outer-test cases,
the held-out metrics are optimistic.

## Decision
Within each outer split, all of these are fit on **outer-train only**:
extraction, rediscovery, lexicon freeze, reliability, weights, teacher, and
the imputer/filter/scaler/model pipeline. Outer-test cases are only
**recoded** with the frozen lexicon (normalized substring) and the fixed
ontology regexes. No LLM is called on test reports.

## Consequences
- Test featurization is not equivalent to extraction. Novel or paraphrased test wording maps to zero (under-featurization, not leakage). This is accepted; do not "fix" it by extracting test reports on the main path.
- Training cases are also recoded by the same rules, so train and test features are produced the same way.
- The ontology regexes are treated as fixed prior knowledge, not learned.
- Hyperparameter search (inner CV) runs on features whose lexicon was built from the whole outer-train set. That slightly optimizes inner-CV scores, but it does not touch outer-test.
