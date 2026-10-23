# 0005: High/low dispersion threshold of 85

Status: Accepted; rationale not documented

## Context
A binary high/low dispersion target is needed alongside the continuous score.

## Decision
High = `dispersion_invasive_DCIS_geographic >= 85`
(`approach1/config.py:DISPERSION_HIGH_THRESHOLD`,
`approach2/config.py:DISPERSION_TRUE_HIGH_THRESHOLD`). Approach 1 also requires
the LLM's binary label to equal `score >= 85`.

## Consequences
- Binary metrics depend on this cutoff. The `metrics_audit` README lists the arbitrary threshold as a moderate limitation.
- The value is duplicated in two constants and in prompt text ("cutoff 85"). Changing it means editing all of these and invalidating checkpoints.
- Why 85 **(unknown)**: possibly clinical or distribution-based. Record the reason here if known.
