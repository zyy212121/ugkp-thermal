"""Opt-in actual CUDA regression; compilation alone does not establish GPU results."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("app", ["CHT", "FSH"])
def test_pr14_fast_actual_cold_initialization_cuda(tmp_path, app):
    if os.environ.get("PR14_FAST_GPU_TESTS") != "1":
        pytest.skip("set PR14_FAST_GPU_TESTS=1 on the real CUDA GPU test host")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/fixtures/pr14_fast_cold/run.py"),
         str(ROOT), str(tmp_path), app],
        text=True, capture_output=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS actual CUDA cold-wall initialization and owner lifetime" in result.stdout
