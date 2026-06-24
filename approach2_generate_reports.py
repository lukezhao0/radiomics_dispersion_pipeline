#!/usr/bin/env python3
"""Generate Approach 2 HTML review reports from saved run artifacts.

Usage:
    python approach2_generate_reports.py --run-dir ./outputs/approach2
    python -m approach2.report_cli --run-dir ./outputs/approach2 --csv-path /path/to/cases.csv --force
"""

from __future__ import annotations

from approach2.report_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
