# Previous report-trimming investigation

What the scripts in `aux/` actually do to MRI and pathology report text, read from the source on 2026-10-22.

This note explains the rules. It does not report how much text those rules removed from the real cohort. The scripts were not run, and no data file was opened.

## The short version

Two different scripts sit in `aux/`. They are not wired together, and the modeling code never calls either one.

| Script | What it changes | What it leaves alone |
|---|---|---|
| `aux/filter_data.py` | Which **columns** are copied into a new CSV | Every cell's text, and every row |
| `aux/trim.py` | The **inside** of two text columns, `preop_MRI_text` and `path_report_text` | Every other column, and every row |

`trim.py` is the script that cuts report text. It treats a report as a stack of blocks separated by blank lines. It looks only at the first line of each block. If that line starts with one of a short list of headings, the whole block is dropped. Every other block is kept, including the words inside it.

`filter_data.py` never cuts a report. It copies 11 named columns from one spreadsheet to another and discards the other columns.

After either script, the modeling pipeline pastes whatever text is in the CSV into the prompt. `shorten_for_prompt` in `approach1/text_utils.py` returns that text unchanged.

## How this was checked

Read in full:

- `aux/trim.py`
- `aux/filter_data.py`
- `aux/check_scripts/check.py` and its README
- The docstrings and column lists of the other `aux/` scripts
- The loaders and the one function that could have shortened a prompt: `approach1/data.py`, `approach1/text_utils.py`, `approach1/prompts/templates.py`, `approach2/eval_data.py`, `approach2/extraction/text_helpers.py`, and the default CSV paths in `approach1/config.py` and `approach2/extraction/config.py`

A search of `*.py` shows `trim_mri_text`, `trim_path_text`, and `filter_data` are mentioned only inside those two scripts. Nothing imports them. `tests/` has no test for them.

`git log` has no commit for `aux/trim.py` or `aux/filter_data.py`. Both files are untracked. There is no commit message in this repo that says they were run, or what the character counts were.

The paths below are the strings written in the source. They point outside this working directory. They were not opened.

## What "a section" means in `trim.py`

A clinical report is a long string in one spreadsheet cell. Radiologists and pathologists break that string into labeled parts: a reason for the study, how the scan was done, what was seen, and a short conclusion.

`trim.py` does not understand that structure. It uses one rule, in `_trim_by_sections`:

1. Turn Windows and old Mac line endings into `\n`.
2. Delete spaces and tabs at the end of each line.
3. Split the cell on a **blank line**. A blank line is a newline, optional spaces or tabs, then at least one more newline (`\n[ \t]*\n+`). A normal single line break does not split anything.
4. For each block, read the first non-empty line and turn it into a heading (the steps are below).
5. If that heading starts with a phrase on the remove list, drop the whole block.
6. Join the surviving blocks with exactly one blank line between them.

There is no memory of a parent heading. A blank line starts a new block, even in the middle of what a person would call the findings. A heading that is not on its own block, with a blank line before it, is invisible to the script.

### How the first line becomes a heading

`_extract_header` does this, in order:

1. Take the first non-empty line of the block.
2. Keep only the text before the first colon. `FINDINGS: mass in the left breast` becomes `FINDINGS`.
3. If that piece contains ` - ` (space, hyphen, space), keep only the text before it. A tab is treated the same way.
4. Strip spaces and the characters `. , ;` from the **end** of that piece.
5. Uppercase it. `Findings`, `findings`, and `FINDINGS` all become `FINDINGS`.
6. If that string is longer than 80 characters, throw the heading away and **keep the block**. The comment in the code says a long first line looks like a sentence, so the script refuses to treat it as a heading.

Matching is `str.startswith` against the uppercase heading (`_match_any_prefix`). The heading must **begin with** the phrase. `INDICATIONS` matches `INDICATION`. `CLINICAL INDICATION` does not, because it begins with `CLINICAL`. The comparison is case-insensitive only because the heading was uppercased. The body of the block is never searched.

## What `trim.py` deletes

The script reads one hardcoded CSV and writes another. Both paths are outside this repo:

- Input: `/Users/lukezhao/projects/onc/PROCESSED_path_cases_MRI_status.csv`
- Output: `/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv`

`main` reads every cell as a string (`dtype=str`, `na_filter=False`), so the letters `NA` stay the letters `NA`. Pandas does not turn them into a missing value. The script requires `preop_MRI_text` and `path_report_text`. The comment at the top also names `dispersion_invasive_DCIS_geographic` and `relapse`, but the code does not check those two. Any other column is copied along. No row is deleted. The two text columns are replaced with the trimmed strings. The script prints the total number of characters in each text column before and after. It does not write those totals to a file.

Empty input stays empty. `None`, a floating-point NaN, and a cell that is only whitespace all become `""`. A cell whose text is the word `NA` is not empty: its heading is `NA`, which is on neither remove list, so the word stays. Later, the modeling code treats `na` as "no report" (`has_report_text` in Approach 1, `_is_missing_text` in Approach 2). That second decision belongs to the pipeline, not to `trim.py`.

### MRI column: `preop_MRI_text`

`MRI_REMOVE_PREFIXES`, applied by `trim_mri_text`:

| Phrase the heading must start with | Examples that match | Examples that do not match |
|---|---|---|
| `INDICATION` | `INDICATION`, `INDICATIONS`, `INDICATION FOR STUDY` | `CLINICAL INDICATION`, `HISTORY` |
| `COMPARISON` | `COMPARISON`, `COMPARISONS` | `COMPARED WITH`, `PRIOR STUDIES` |
| `PROCEDURE COMMENTS` | `PROCEDURE COMMENTS` | |
| `PROCEDURE` | `PROCEDURE`, `PROCEDURES`, and also `PROCEDURE COMMENTS` | `PROTOCOL` |
| `TECHNIQUE` | `TECHNIQUE`, `TECHNIQUES` | `TECHNICAL FACTORS` |
| `DATE` | `DATE`, `DATE OF SERVICE`, `DATED` | `STUDY DATE` |
| `CONTRAST` | `CONTRAST`, `CONTRAST ADMINISTERED` | `WITH CONTRAST` |
| `REFERENCE` | `REFERENCE`, `REFERENCES` | `RECOMMENDATION` |

`PROCEDURE COMMENTS` is listed before `PROCEDURE`. Both would match a heading of `PROCEDURE COMMENTS`, because matching is "starts with," so the extra entry does not change the result.

The comment at the top says to keep `FINDINGS` (any capitalization), `IMPRESSION`, and `ENHANCEMENT PATTERN`. There is no keep-list in the code. Those headings survive because they do not start with a phrase on the remove list. The same is true of every other heading the list does not name: `HISTORY`, `CLINICAL HISTORY`, `PROTOCOL`, `RECOMMENDATION`, `ADDENDUM`, `BI-RADS`, and anything similar.

`DATE` is four letters. A block whose first line is a short sentence beginning with "Date …", with no colon and 80 characters or fewer, is removed, because the whole line becomes the heading and that heading starts with `DATE`. The same sentence longer than 80 characters is kept, because the script then discards the heading.

`diagnostic_MRI_text` is never passed to `trim_mri_text`. A diagnostic MRI report in that column would be copied in full.

### Pathology column: `path_report_text`

`PATH_REMOVE_PREFIXES`, applied by `trim_path_text`:

| Phrase the heading must start with | Examples that match | Examples that do not match |
|---|---|---|
| `ACCESSION NO` | `ACCESSION NO`, `ACCESSION NO.` | `ACCESSION` alone, `ACCESSION NUMBER` (`NUMBER` diverges at `U` versus `O`) |
| `OPERATION` | `OPERATION`, `OPERATIONS`, `OPERATION PERFORMED` | `OPERATIVE PROCEDURE` (`OPERATIVE` diverges from `OPERATION`) |
| `BLOCK KEY` | `BLOCK KEY`, `BLOCK KEYS`, `BLOCK KEY - A1` | `BLOCK SUMMARY`, `SPECIMEN` |
| `REFERENCE` | `REFERENCE`, `REFERENCES` | `RECOMMENDATION` |

Pathology headings that stay, because they are not on this list, include `FINAL DIAGNOSIS`, `GROSS DESCRIPTION`, `MICROSCOPIC DESCRIPTION`, `SYNOPTIC`, `COMMENT`, `CLINICAL HISTORY`, and `DIAGNOSIS`. An MRI-style `COMPARISON` or `TECHNIQUE` heading is also kept if it appears in the pathology column. The two columns do not share a remove list.

The comment's phrase "Accession No..." is this prefix rule. A heading that is only `Accession:` becomes `ACCESSION` and is kept.

### A made-up walk-through

The report below is invented to show the rules. It is not from the cohort.

```text
INDICATION:
Screening follow-up.

TECHNIQUE:
Standard protocol.

FINDINGS:
There is a mass.

Comparison with the prior exam shows it is smaller.

IMPRESSION:
Residual mass.
```

Blank lines split this into five blocks.

| Block's first line | Heading | Result |
|---|---|---|
| `INDICATION:` | `INDICATION` | Dropped |
| `TECHNIQUE:` | `TECHNIQUE` | Dropped |
| `FINDINGS:` | `FINDINGS` | Kept, including the sentence on the next line of the **same** block |
| `Comparison with the prior exam shows it is smaller.` | that whole line, uppercased | Dropped. It is its own block, it is under 80 characters, and it starts with `COMPARISON` |
| `IMPRESSION:` | `IMPRESSION` | Kept |

What remains:

```text
FINDINGS:
There is a mass.

IMPRESSION:
Residual mass.
```

The sentence about the prior exam was inside the findings a person would read, and it was still deleted. The script never asks which heading a block "belongs to." It only looks at that block's first line.

Now suppose the same words had no blank lines at all, and the cell started with `INDICATION:`. The whole cell would be one block. The heading would be `INDICATION`. Findings and impression would be deleted with it.

Suppose instead the cell started with `FINDINGS:` and `INDICATION:` appeared later on its own line with no blank line in front of it. Nothing would be deleted. The indication line would be ordinary text inside the findings block.

### What else the trim changes, besides deleted blocks

Surviving text is joined with exactly one blank line. Several blank lines in a row become one. Spaces at the ends of lines are removed. The cell is stripped at both ends. The words inside a kept block are not edited: there is no word list, no sentence splitter, and no character limit on a block that was kept.

The output file is written by pandas `to_csv`. Untouched columns keep the strings that were read. Quoting and line endings follow pandas defaults, so the new file need not be a byte-for-byte copy of the input even in columns the trimmer did not edit.

## What `filter_data.py` does

`filter_data.py` is a column copier. The usage line is:

```text
python filter_data.py INPUT.csv OUTPUT.csv
```

There is no default path. The input is decoded as UTF-8 with a byte-order mark allowed (`utf-8-sig`). The output is UTF-8 with `\n` line endings. Cell text is copied as the `csv` module read it. Nothing is stripped, lowercased, or cut.

The output columns, in this order, are the list `COLUMNS`:

| Column | What it holds, from the name and from how this repo uses it |
|---|---|
| `case_id` | Case identifier. The modeling loaders use it when it is present. |
| `diagnostic_MRI_text` | A diagnostic MRI report. `trim.py` does not edit this column. No modeling loader reads it. |
| `preop_MRI_text` | The post-treatment, pre-surgery MRI report. This is the MRI text the models read, and the column `trim.py` trims. |
| `path_report_text` | The pathology report. This is the pathology text the models read, and the column `trim.py` trims. |
| `relapse` | Relapse label. The models read it. |
| `RFS` | Named in this list only. No other file in this repo mentions it. |
| `dmax_invasive` | Named in this list only. No other file in this repo mentions it. |
| `dispersion_invasive_DCIS_geographic` | The dispersion score. The models read it. |
| `index_side` | Which breast the labels refer to. The models read it. |
| `RCB_Continuous` | Named in this list only. No other file in this repo mentions it. |
| `RCB_Class` | Named in this list only. No other file in this repo mentions it. |

If any of those 11 names is missing, or if a column name is duplicated, the script stops and writes nothing. Rows stay in their original order. A short row is padded with `""` for the missing fields. Columns that are not on the list are left out of the new file.

The modeling loaders require a smaller set: `preop_MRI_text`, `path_report_text`, `index_side`, `dispersion_invasive_DCIS_geographic`, and `relapse`, plus optional `case_id` (`approach1/data.py`). `filter_data.py` keeps five further columns that this repository's model code never reads.

This repository does not show `filter_data.py` being run before or after `trim.py`. `trim.py`'s input and output names are fixed. `filter_data.py`'s paths are whatever the person typed. A person could have used one script's output as the other script's input. The source does not record that.

## What the modeling code does with the text afterward

Both approaches default to the trimmed filename:

- `approach1/config.py`: `CSV_PATH = "/Users/lukezhao/projects/onc/PROCESSED_TRIMMED_path_cases_MRI_status.csv"`
- `approach2/extraction/config.py`: the same string

That is the same string as `trim.py`'s `OUT_PATH`. The usage comments in `approach1.py` and `approach2.py` name a different string, with an extra `data/` folder: `/Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv`. Example commands in `aux/llm_token_audit.py` and `aux/pathology_underperformance_audit.py` use the path **without** `data/`. Which file a given run actually opened is whichever path was passed that day. This investigation did not look at either file.

Once a CSV is loaded, report text is copied into the case and sent on:

- Approach 1 builds the prompt in `report_fields_for_prompt` (`approach1/prompts/templates.py`). It calls `shorten_for_prompt`, and that function returns its input unchanged (`approach1/text_utils.py`).
- A tier can withhold a **whole** report. In the MRI-only tier the pathology field is replaced with the literal `<NOT PROVIDED IN THIS TIER>`, and the pathology-only tier does the same for the MRI field. The report that is supplied is supplied in full.
- Approach 1 skips a case on an MRI tier when `preop_MRI_text` is empty or a placeholder (`missing_preop_MRI_text_required_for_this_tier` in `approach1/splits.py`). Placeholders include `na`, `n/a`, `nan`, `none`, `null`, `missing`, `<na>`, `not available`, and `not applicable`.
- Approach 2 drops rows that have no numeric dispersion score (`get_target_frame` in `approach2/eval_data.py`). For datasets that need MRI (`mri`, `combined`, and the pathology-calibration and teacher-student keys), it also drops cases whose `preop_MRI_text` is empty or a placeholder (`filter_missing_mri_for_dataset`). Pathology-only runs keep those cases. Approach 2's placeholder list adds `[missing]`, `no mri`, and `no report`, and does not include `<na>`.

If `trim.py` reduced an MRI cell to an empty string, the later MRI pathways would treat that case as missing MRI. That follows from reading the two pieces of code. It is not a count of cases.

## The other files in `aux/`

None of these cut text out of a report.

| File | What it is |
|---|---|
| `aux/check_scripts/check.py` | Compares two CSVs row by row, in order, on five non-text columns: `case_id`, `diagnostic_MRI_exists`, `preop_MRI_exists`, `relapse`, `dispersion_invasive_DCIS_geographic`. The word "trim" in its `normalize` function means collapsing whitespace so `1` and `1 ` can match. It does not edit a file. Default paths, not opened: `data/UPDATED_path_MRI_cases.csv` and `data/path_cases_MRI_status.csv` under `/Users/lukezhao/projects/onc/`. The README next to it describes an older, smaller set of flags than the script now has (`--encoding`, `--no-normalize`). |
| `aux/pathology_underperformance_audit.py` | Scores completed Approach 2 outputs. It searches pathology text for words such as `findings`, `impression`, `gross`, and `microscopic`, and records yes/no flags. It does not delete those parts. Its example command points at the trimmed CSV. It also looks for a text column under several names, including `trimmed_pathology_text` and `trimmed_path_text`. `trim.py` does not create those names. It keeps the column name `path_report_text`. |
| `aux/compare.py` | Compares an MRI audit table with a pathology audit table that already exist. |
| `aux/plot_mri_audit_high_low.py` | Plots MRI audit metrics for high versus low dispersion. |
| `aux/llm_token_audit.py` | Estimates tokens and cost from saved extractions and from report text as stored. Its example command points at the trimmed CSV. It does not shorten that text. |
| `aux/plot_llm_token_costs.py` | Plots a cost table that the token audit already wrote. |
| `aux/plot_n_features_by_model.py` | Bar chart of feature counts from a hyperparameters CSV. |
| `aux/regenerate_two_new_plots.py` | Redraws AUROC and Spearman charts from a metrics summary. |
| `aux/plot_case_feature_heatmap.py` | Heatmaps of case-by-feature matrices. |
| `aux/concentric_circles.py` | A diagram of nested split sizes (full data, 80% outer train, 64% inner train). No cohort data. |

## What this note does not know

- How many characters, sections, or cases the trimmer changed. The script would print two totals to the terminal. Those totals are not stored in this repo, and the script was not run.
- Whether the real reports put a blank line between headings. The rules above are what the code does with whatever string it is given. If a real report puts every heading in one block, the first line decides the fate of the entire report.
- Whether anyone ran `filter_data.py`, `trim.py`, or both, and in which order. The filenames in the source are consistent with `trim.py` having produced the CSV the models default to. Consistency is not a run log.
- What is inside the CSVs named above. They were not opened. Report text can contain identifiers, so they stay closed.

## Where the rules live

| Rule | Place |
|---|---|
| Blank-line split, heading cleanup, 80-character guard, prefix match | `aux/trim.py`: `_normalize_newlines`, `_extract_header`, `_match_any_prefix`, `_trim_by_sections` |
| MRI remove list | `aux/trim.py`: `MRI_REMOVE_PREFIXES`, `trim_mri_text` |
| Pathology remove list | `aux/trim.py`: `PATH_REMOVE_PREFIXES`, `trim_path_text` |
| Read one CSV, write the trimmed CSV, print character totals | `aux/trim.py`: `main` |
| Copy 11 columns, leave cell text unchanged | `aux/filter_data.py`: `COLUMNS`, `main` |
| Prompt text is the cell, unchanged | `approach1/text_utils.py`: `shorten_for_prompt` |
| A whole modality withheld, or a case skipped for missing MRI | `approach1/prompts/templates.py`: `report_fields_for_prompt`; `approach1/splits.py` |
| Cases dropped for a missing score, or for missing MRI on an MRI pathway | `approach2/eval_data.py`: `get_target_frame`, `filter_missing_mri_for_dataset` |
