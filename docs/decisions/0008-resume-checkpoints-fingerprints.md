# 0008: Resumable runs guarded by fingerprints

Status: Accepted

## Context
Runs make hundreds of paid API calls and can be interrupted. Silently
reusing results from a different configuration would corrupt metrics.

## Decision
- **Approach 1:** per-case JSONL checkpoints, plus a per-config `_resume_checkpoint/COMPLETED.json`. The fingerprint covers the shot set, rows, tier, test row indices, skipped rows, and `RESUME_SCRIPT_VERSION` (`approach1-3-v2`).
- **Approach 2:** per-case extraction checkpoints, plus a per-split `COMPLETED.json` with a fingerprint over CSV path, outer scheme/repeats/frac/folds, seed, stability threshold, feature cap, modalities, representations, model, reasoning effort, and `RESUME_SCRIPT_VERSION` (2).
- Mismatch means recompute.

## Consequences
- Prompt or schema edits are **not** in the Approach 1 fingerprint. Bump `RESUME_SCRIPT_VERSION` by hand when they change, or stale predictions will be reused.
- Approach 1 fingerprints use row numbers, not case IDs, so a re-sorted CSV would not be detected.
- The A2 fingerprint stores the CSV *path*, not a content hash.
