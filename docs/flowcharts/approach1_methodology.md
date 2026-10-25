# How Approach 1 works: a one-page diagram

Written 2026-10-24 for non-technical collaborators, from reading the source code
only. Nothing was run, and no patient data was looked at. All example wording is
taken from the prompt code or invented.

**In two sentences.** We show a large language model (LLM) four solved example
cases, then give it one new patient's reports and ask it to predict how
scattered the remaining tumor is, and whether the cancer will come back. We
check each answer automatically and compare it with what actually happened.

Read left to right. The colored box at the top of each column is the step. The
box underneath explains how that step is done.

```mermaid
flowchart LR
    subgraph S1["Step 1"]
        direction TB
        A1["<b>Raw MRI and<br/>pathology reports</b>"]
        A1d["One row per patient's treated breast:<br/>• MRI report, after chemotherapy<br/>and before surgery<br/>• pathology report<br/>• which breast (left or right)<br/>the answers refer to<br/>• the true dispersion score and<br/>whether the cancer came back"]
        A1 --- A1d
    end

    subgraph S2["Step 2"]
        direction TB
        A2["<b>Trimmed reports</b><br/>boilerplate removed"]
        A2d["Each report is split at blank lines.<br/>A block is dropped when its first line<br/>starts with a listed heading.<br/><br/><b>MRI</b>: drops reason for exam, comparison,<br/>procedure/technique, contrast, dates,<br/>references. Findings and impression stay.<br/><br/><b>Pathology</b>: drops accession number,<br/>operation, block key, references.<br/>Diagnosis and descriptions stay.<br/><br/>Everything else is kept word for word."]
        A2 --- A2d
    end

    subgraph S3["Step 3"]
        direction TB
        A3["<b>Set up the experiment</b>"]
        A3d["• Pick 4 solved example cases:<br/>2 high and 2 low dispersion<br/>• Try 2 different example sets, to see<br/>whether the choice of examples matters<br/>• Run each set 3 ways: MRI only,<br/>pathology only, MRI + pathology<br/>• Every other case is a test case.<br/>Cases with no MRI report sit out<br/>the MRI versions<br/><br/>2 example sets × 3 ways = 6 runs"]
        A3 --- A3d
    end

    subgraph S4["Step 4"]
        direction TB
        A4["<b>LLM reads each test case<br/>and predicts</b><br/>one prompt per case, sent to a GPT-5<br/>model in Stanford's secure AI sandbox"]
        subgraph PROMPT["What the prompt contains, top to bottom"]
            direction TB
            P1["<b>1. Role and ground rules</b><br/>You are a careful clinical reader. Use only<br/>the report text given, don't make anything up,<br/>and reply in a fixed format."]
            P2["<b>2. The task</b><br/>Focus on the indicated breast. Predict:<br/>dispersion score (0 to 450) · high or low<br/>(high = 85 or more) · relapse yes/no ·<br/>up to 6 exact quotes as evidence ·<br/>a short step-by-step rationale"]
            P3["<b>3. Rules</b><br/>Quotes must be copied word for word ·<br/>high/low must match the score · ignore the<br/>other breast · don't guess about a report<br/>that wasn't provided"]
            P4["<b>4. Hints: what dispersion tends to sound like</b><br/>Often high: 'scattered small clusters', patchy or<br/>multifocal residual tumor, non-mass enhancement,<br/>close or positive margins, extensive DCIS<br/>Often low: one well-circumscribed mass,<br/>near-complete response, clear margins"]
            P5["<b>5. Answer form</b><br/>A fill-in-the-blanks template (JSON): score,<br/>high/low, relapse, quotes, summary, rationale"]
            P6["<b>6. Four worked examples</b><br/>Each shows its report(s) for this run, which<br/>breast, and the true answers: score,<br/>high/low, relapse"]
            P7["<b>7. The new case</b><br/>Its report(s) and which breast, with the<br/>answers hidden: 'now predict this case'.<br/>Ends with a code word the model must repeat<br/>back, as an attention check (not evidence)"]
            P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7
        end
        A4 --- PROMPT
    end

    subgraph S5["Step 5"]
        direction TB
        A5["<b>Check every answer</b>"]
        A5d["• Is the answer complete and well formed?<br/>• Is the score between 0 and 450?<br/>• Does each quote really appear in the report?<br/>Near-misses are corrected to the exact text;<br/>the rest are removed<br/>• Does high/low agree with the score?<br/>• Did it repeat the code word?<br/>(kept as an attention score)<br/><br/>If a check fails, ask again, up to 3 tries"]
        A5 --- A5d
    end

    subgraph S6["Step 6"]
        direction TB
        A6["<b>Compare with the<br/>true answers</b>"]
        A6d["• Dispersion score: how far off on average,<br/>and does it put cases in the right order?<br/>• High vs low: how often right, and how well<br/>the predicted score separates high from low<br/>• Relapse: how often right. The model only<br/>says yes/no, not a risk level, so<br/>ranking-based scores mean less here<br/>• Each number gets an uncertainty range<br/>(bootstrap resampling of the cases)"]
        A6 --- A6d
    end

    subgraph S7["Step 7"]
        direction TB
        A7["<b>Results report</b>"]
        A7d["• Score tables for all 6 runs<br/>• Charts: predicted vs true, confusion<br/>tables, ROC and precision-recall curves<br/>• Which quoted phrases the model<br/>leaned on most<br/>• Cost of the LLM calls<br/><br/>Can be rebuilt from saved answers<br/>without asking the LLM again"]
        A7 --- A7d
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7

    classDef step fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0f172a
    classDef detail fill:#f8fafc,stroke:#94a3b8,color:#0f172a
    classDef prompt fill:#fef9c3,stroke:#a16207,color:#0f172a
    class A1,A2,A3,A4,A5,A6,A7 step
    class A1d,A2d,A3d,A5d,A6d,A7d detail
    class P1,P2,P3,P4,P5,P6,P7 prompt
```

## Plain-text version

The same diagram, for viewers that do not render Mermaid.

```text
[ 1. Raw MRI + pathology reports ]
    |  one row per patient's treated breast: MRI report (after chemo, before
    |  surgery), pathology report, which breast, true dispersion score, relapse
    v
[ 2. Trimmed reports ]
    |  split at blank lines; drop a block whose first line is a listed heading
    |  MRI: drop reason for exam, comparison, procedure/technique, contrast,
    |       dates, references. Keep findings and impression.
    |  Pathology: drop accession number, operation, block key, references.
    |       Keep diagnosis and descriptions. Everything else is kept as written.
    v
[ 3. Set up the experiment ]
    |  4 solved examples (2 high, 2 low) x 2 example sets
    |  x 3 ways (MRI only / pathology only / MRI + pathology) = 6 runs
    |  every other case is a test case; no-MRI cases sit out the MRI versions
    v
[ 4. LLM reads each test case and predicts ]   (one prompt per case)
    |    [ Role and ground rules: careful clinical reader, use only the text ]
    |      -> [ Task: score 0-450, high/low (85+), relapse yes/no, quotes, rationale ]
    |      -> [ Rules: exact quotes, high/low matches score, index breast only ]
    |      -> [ Hints: wording that often means high vs low dispersion ]
    |      -> [ Answer form: fixed JSON template ]
    |      -> [ 4 worked examples: reports, breast, true score/high-low/relapse ]
    |      -> [ New case: reports + breast, answers hidden, "now predict" + code word ]
    v
[ 5. Check every answer ]
    |  complete? score in range? quotes really in the report (fix near-misses,
    |  drop the rest)? high/low agrees with score? code word repeated?
    |  failed check -> ask again, up to 3 tries
    v
[ 6. Compare with the true answers ]
    |  score error and ordering; high/low accuracy and separation;
    |  relapse accuracy (yes/no only); uncertainty ranges by bootstrap
    v
[ 7. Results report ]
       tables, charts, most-cited phrases, LLM cost
```

## What the words mean

- **Dispersion (dispersiveness):** how spread out the remaining tumor is after
  chemotherapy, from one compact lump to many scattered islands. It is measured
  as a score; 85 or more counts as "high".
- **Relapse:** whether the cancer came back.
- **Few-shot:** showing the model a few solved examples inside the prompt,
  instead of training it on data. The model's internals are never changed.
- **Index breast (index side):** the breast (left or right) the score and relapse
  label belong to. Reports sometimes describe both breasts.
- **Run / version (tier in the code):** which report or reports the model is shown.

## What this picture leaves out

- The model gives one answer per case. Asking again could give a slightly
  different answer.
- All test cases in a run see the same four examples, so results can depend on
  which examples were picked. That is why two example sets are compared.
- The MRI versions skip cases that have no MRI report, so the three versions are
  not scored on exactly the same cases.
- Long runs save every answer as they go, so a stopped run can pick up where it
  left off. This does not change the results.

## Where this lives in the code (for maintainers)

| Step | Files |
|---|---|
| 2. Trimming | `aux/trim.py` (`MRI_REMOVE_PREFIXES`, `PATH_REMOVE_PREFIXES`, `_trim_by_sections`) |
| 1, 3. Loading, example sets, runs | `approach1/data.py`, `approach1/config.py` (`SHOT_SETS`, `MODALITY_TIERS`), `approach1/splits.py` |
| 4. Prompt | `approach1/prompts/system.py`, `templates.py`, `descriptors.py`, `tokens.py` |
| 4. LLM call | `approach1/inference.py`, `approach1/api/client.py` |
| 5. Checking | `approach1/schema/prediction.py` |
| 6–7. Scoring and report | `approach1/evaluation/` (`runner.py`, `bootstrap.py`, `results_report.py`), `approach1/orchestration.py` |

For the technical version of this flow, see `docs/PIPELINE_GUIDE.md` §4 and
`docs/ARCHITECTURE.md` §3.
