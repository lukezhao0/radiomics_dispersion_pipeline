# 0003: Repeated Monte Carlo outer splits with mean deduplication

Status: Accepted (default kept deliberately; see `approach2_progress.md` 2026-06-23 "Outer split strategy")

## Context
Approach 2 needs outer resampling for an honest estimate. Options:
stratified k-fold (each case tested once) or repeated Monte Carlo (MC)
random splits.

## Decision
Default `--outer-scheme repeated_mc`: 5 stratified 80/20 draws (seeds
`RANDOM_SEED + rep`). Cases appearing in several test sets have their
predictions averaged (`deduplicate_outer_predictions`) before pooled metrics.
`stratified_kfold` stays available, and `validate_outer_splits` checks that
each case is held out exactly once under k-fold.

## Consequences
- With 5 × 20% draws, about 1 − 0.8⁵ ≈ 67% of cases are expected to be tested at least once. The rest have no held-out prediction, so pooled metrics cover a subset. (The combined-modality audit reports a smaller Approach 2 N than Approach 1 partly for this reason.)
- Averaged probabilities are smoother than single-split ones, which can affect calibration metrics.
- Reason for keeping repeated MC as default **(inferred)**: per-split variability estimates and continuity with earlier runs.
- Changing the default would change what every reported metric means and invalidate resume fingerprints.
