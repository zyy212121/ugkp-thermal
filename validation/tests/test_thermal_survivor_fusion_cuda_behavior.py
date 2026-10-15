from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT

@pytest.mark.parametrize('branch', ['FSH', 'CHT'])
def test_exact_survivor_pipeline_preserves_thermal_state(tmp_path, branch):
    nvcc = Path("/usr/local/cuda/bin/nvcc")
    smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not nvcc.is_file() or not smi.is_file():
        pytest.skip("CUDA compiler and GPU required")
    if subprocess.run([str(smi), "-L"], capture_output=True).returncode:
        pytest.skip("CUDA GPU unavailable")
    result = subprocess.run(
        ["python3", str((VALIDATION_ROOT / 'tools/tests/fixtures/thermal_workers/fused_pipeline.py')),
         str(ROOT), str(tmp_path), branch],
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS actual survivor bins" in result.stdout
