"""Import and CLI smoke tests for approach2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import approach2
import approach2.config
import approach2.io_atomic
import approach2.metrics
import approach2.models
import approach2.text_utils


def test_package_reexports():
    assert approach2.DISPERSION_TRUE_HIGH_THRESHOLD == 85.0
    assert approach2.ModelSpec is not None
    assert callable(approach2.normalize_text)


def test_approach2_cli_help():
    import os

    pipeline_dir = Path(__file__).resolve().parents[2]
    script = pipeline_dir / "approach2.py"
    env = dict(os.environ)
    env["SANDBOX_API_KEY"] = "dummy-test-key"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(pipeline_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--csv-path" in result.stdout
    assert "--model" in result.stdout
