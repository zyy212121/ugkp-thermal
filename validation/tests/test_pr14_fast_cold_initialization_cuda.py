"""Opt-in actual CUDA regression; compilation alone does not establish GPU results."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = SOURCE_ROOT


@pytest.mark.parametrize("app", ["CHT", "FSH"])
def test_pr14_fast_actual_cold_initialization_cuda(tmp_path, app):
    if os.environ.get("PR14_FAST_GPU_TESTS") != "1":
        pytest.skip("set PR14_FAST_GPU_TESTS=1 on the real CUDA GPU test host")
    result = subprocess.run(
        [sys.executable, str((VALIDATION_ROOT / 'tools/tests/fixtures/pr14_fast_cold/run.py')),
         str(ROOT), str(tmp_path), app],
        text=True, capture_output=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS actual CUDA cold-wall initialization and owner lifetime" in result.stdout
