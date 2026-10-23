# check.py

Verifies that two CSV files contain identical data, row for row and in the same
order, across a fixed set of columns.

## Columns compared

- `case_id`
- `diagnostic_MRI_exists`
- `preop_MRI_exists`
- `relapse`
- `dispersion_invasive_DCIS_geographic`

Only these columns are read from each file; everything else is ignored.

## Requirements

- Python 3
- `pandas`

The repo's virtualenv at `/Users/lukezhao/projects/onc/.venv` already has pandas
installed.

## How to run

The two default file paths are already baked into the script, so from the repo
root you can just run:

```bash
/Users/lukezhao/projects/onc/.venv/bin/python aux/check_scripts/check.py
```

That is equivalent to:

```bash
/Users/lukezhao/projects/onc/.venv/bin/python aux/check_scripts/check.py \
  --csv1 /Users/lukezhao/projects/onc/data/UPDATED_path_MRI_cases.csv \
  --csv2 /Users/lukezhao/projects/onc/data/path_cases_MRI_status.csv
```

If you have the virtualenv activated (`source .venv/bin/activate`), plain
`python aux/check_scripts/check.py` works too.

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--csv1` | `data/UPDATED_path_MRI_cases.csv` (absolute path) | First CSV file |
| `--csv2` | `data/path_cases_MRI_status.csv` (absolute path) | Second CSV file |
| `--max-report` | `50` | Maximum number of mismatched rows to print |

## Output

On success:

```
PASS: both files are identical across the compared columns, in the same order.
```

On failure the script prints the row count of each file, how many rows differ,
a per-column count of mismatches, and then each mismatching row with its
0-based row index, its line number in the CSV, and the two conflicting values.

Exit code is `0` when the files match and `1` when they do not, so the script
can be used in a shell pipeline or CI check.

## Comparison details

- Values are read as raw strings (`dtype=str`, `keep_default_na=False`) so that
  pandas does not silently coerce the same value differently in the two files
  (for example `1` in one file and `1.0` in the other).
- Leading and trailing whitespace is stripped before comparing.
- Comparison is positional: row *n* of the first file is compared against row
  *n* of the second file. Rows are never sorted or matched up by `case_id`, so a
  reordering of the files will be reported as a mismatch.
- If the two files have different row counts, that is reported and the first
  *min(len1, len2)* rows are still compared.
- If a required column is missing from a file, the script exits with an error.
