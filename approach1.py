#!/usr/bin/env python3
"""CLI entry point: python approach1.py ...

Usage:
python pipeline/approach1.py \
    --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
    --outdir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach1_pipeline_062426

python pipeline/approach1.py \
    --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
    --outdir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach1_pipeline_062426 \
    --model gpt-5-nano \
    --reasoning-effort low

python pipeline/approach1.py \
    --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
    --outdir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach1_pipeline_062526 \
    --model gpt-5 \
    --reasoning-effort medium

"""

from approach1.cli import main

if __name__ == "__main__":
    main()
