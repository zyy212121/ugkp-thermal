from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("branch,bits", [("FSH", 64), ("CHT", 32), ("CHT", 64)])
def test_shared_s1_l2_survivor_pipeline_preserves_thermal_state(tmp_path, branch, bits):
    nvcc = Path("/usr/local/cuda/bin/nvcc")
    smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not nvcc.is_file() or not smi.is_file():
        pytest.skip("CUDA compiler and GPU required")
    if subprocess.run([str(smi), "-L"], capture_output=True).returncode:
        pytest.skip("CUDA GPU unavailable")
    result = subprocess.run(
        ["python3", str(ROOT / "tests/fixtures/thermal_workers/s1_pipeline.py"),
         str(ROOT), str(tmp_path), branch, str(bits), "S1"],
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS actual survivor bins" in result.stdout
