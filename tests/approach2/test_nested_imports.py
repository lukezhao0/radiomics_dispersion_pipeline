"""Import smoke tests for modularized approach2 nested evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_nested_eval_modules_import():
    import approach2.audit
    import approach2.calibration
    import approach2.cli
    import approach2.eval_data
    import approach2.lexicon
    import approach2.models_ml
    import approach2.orchestration
    import approach2.recoding
    import approach2.reports
    import approach2.splits
    import approach2.features.normalize
    import approach2.features.matrices

    assert approach2.cli.main is not None


def test_approach2_aux_help():
    import os

    pipeline_dir = Path(__file__).resolve().parents[2]
    script = pipeline_dir / "approach2_aux.py"
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
    assert "--report-mode" in result.stdout
