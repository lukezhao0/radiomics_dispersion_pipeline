# 0010: Stability threshold default (unresolved)

Status: **Unresolved**

## Context
The stability threshold decides which LLM-proposed phrases become features
(selection frequency across rediscovery subsplits). History in
`approach2_progress.md`: 0.60, then 0.35, then a constant of 0.20.

## Current state (verified in code 2026-10-22)
- `approach2/config.py`: `DEFAULT_STABILITY_THRESHOLD = 0.20`.
- `approach2/cli.py`: `--stability-threshold` `default=0.60`. The help text says "USE DEFAULT_STABILITY_THRESHOLD FOR 0.20, RIGHT NOW HARD CODED 0.60".
- `docs/approach2_feature_stability_and_selection.md` describes the default as 0.20.

## Decision needed
Choose one default and make the argparse default reference the constant
(or vice versa).

## Consequences of changing it
- It changes which phrase features exist, and so the `phrase_binary` results.
- It is part of the resume fingerprint, so completed splits would be recomputed.
- Past results should be labeled with the threshold actually used. Read each run's `*_stable_lexicon_metadata_*.json` (`stability_threshold`) or `run.log`.
