# 0006: Verbatim quote grounding, with repair or drop

Status: Accepted (2026-06-24 entries in both progress logs)

## Context
LLM rationales and extracted phrases must be auditable against the source
report. Models sometimes paraphrase or over-quote, which used to cause
whole-response failures.

## Decision
- Quotes (`key_evidence` in A1, phrase `quote` in A2) must be exact substrings of the supplied report text: ≤ 35 words hard cap, 25 targeted in the prompt.
- A non-matching or overlong quote is repaired to the closest exact report span when possible, otherwise dropped with a warning (`approach1/schema/`, `approach2/extraction/schema.py`).
- A dropped quote does not fail the whole response.

## Consequences
- Outputs contain real report text. Run directories and logs must be treated as PHI.
- Repair can substitute a span the model did not literally write. Warnings and counts are recorded (`n_repaired_key_evidence_quotes`, `n_dropped_key_evidence_quotes`).
- The A2 extraction prompt separately says "≤ 30 words", a minor inconsistency.
