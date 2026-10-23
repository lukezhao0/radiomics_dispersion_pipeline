# Architecture Decision Records

One file per significant design choice, in the format Context / Decision /
Consequences. Most were reconstructed on 2026-10-22 from code and commit
history. Reasons marked **(inferred)** were not written down by the authors.
Add new ADRs with the next number. Supersede old ones by adding a new ADR and
marking the old one `Superseded by NNNN`; don't delete them.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-two-parallel-approaches.md) | Run two approaches: LLM-as-predictor and LLM-as-feature-proposer | Accepted |
| [0002](0002-train-only-leakage-boundary.md) | Everything learned in Approach 2 uses outer-train only; test is recoded by rules | Accepted |
| [0003](0003-repeated-mc-outer-splits-with-dedup.md) | Default outer scheme is repeated Monte Carlo with mean deduplication | Accepted |
| [0004](0004-missing-mri-is-exclusion.md) | Missing MRI excludes a case from MRI pathways; never zero-filled | Accepted |
| [0005](0005-high-low-threshold-85.md) | High/low dispersion cutoff is 85 | Accepted (rationale undocumented) |
| [0006](0006-verbatim-quote-grounding.md) | LLM outputs must cite verbatim quotes; near-misses repaired or dropped | Accepted |
| [0007](0007-no-temperature-reasoning-effort.md) | No `temperature` in API payloads; `reasoning_effort` configurable | Accepted |
| [0008](0008-resume-checkpoints-fingerprints.md) | Resumable runs guarded by config fingerprints | Accepted |
| [0009](0009-packages-with-thin-cli-shims.md) | Logic in packages, top-level scripts are thin shims | Accepted |
| [0010](0010-stability-threshold-default.md) | Stability threshold default | **Unresolved** |
| [0011](0011-prespecified-audit-headline-models.md) | Combined-modality audit uses pre-specified headline models | Accepted |
