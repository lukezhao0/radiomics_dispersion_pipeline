#!/usr/bin/env python3
"""
Thin CLI wrapper and backward-compatible entry point for approach2 nested evaluation.

Implementation lives in the approach2 package. This module preserves the original
script entry point:

Usage:

python pipeline/approach2.py \
  --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach2_pipeline \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary

python pipeline/approach2.py \
  --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach2_pipeline_062426 \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary
  --model gpt-5-nano \
  --reasoning-effort low


python pipeline/approach2.py \
  --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach2_pipeline \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary \
  --max-api-workers 2 \
  --parallel-modality-workers 2 \
  --ml-n-jobs 1


python pipeline/approach2.py \
  --csv-path /Users/lukezhao/projects/onc/data/PROCESSED_TRIMMED_path_cases_MRI_status.csv \
  --out_dir /Users/lukezhao/projects/onc/sabcs/securegpt_dispersion_approach2_pipeline \
  --enable-pathology-calibration \
  --enable-teacher-student \
  --modalities mri path combined \
  --representations group_binary group_count group_status phrase_binary \
  --max-api-workers 2 \
  --parallel-modality-workers 2 \
  --ml-n-jobs 1 \
  --yes
"""

from __future__ import annotations

from approach2.cli import main

if __name__ == "__main__":
    main()
