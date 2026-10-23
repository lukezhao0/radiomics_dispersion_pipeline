# 0007: No `temperature`; configurable `reasoning_effort`

Status: Accepted (commit `d4dc463` "remove temperature implementation"; 2026-06-24 progress entries)

## Context
The Sandbox GPT-5 deployments did not accept non-default `temperature`
values **(inferred from the removal)**. GPT-5 models expose `reasoning_effort`
instead.

## Decision
API payloads omit `temperature`. `--reasoning-effort` takes
`minimal|low|medium|high|none` (default `medium`; `none` omits the field), via
`common/reasoning_effort.py`. Reasoning tokens are billed as output.

## Consequences
- Predictions are not deterministic, and reruns can differ. There is no repeated sampling.
- Reasoning effort is part of the Approach 2 resume fingerprint and should be reported with every result.
- Do not reintroduce `temperature`.
