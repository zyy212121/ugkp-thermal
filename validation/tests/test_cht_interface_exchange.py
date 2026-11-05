"""Exercise the production CUDA interface exchange against the original CPU order."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from validation.tools.paths import SOURCE_ROOT


def test_cht_interface_exchange(tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file():
        pytest.skip("CUDA compiler unavailable")
    executable = tmp_path / "interface_exchange"
    command = [
        nvcc, "-std=c++17", "-O3", "--fmad=false",
        "-arch=" + os.environ.get("UGKWP_CUDA_ARCH", "sm_89"),
        str(SOURCE_ROOT / "validation/resident_execution/test_cht_interface_exchange.cu"),
        str(SOURCE_ROOT / "applications/CHT/gpu/GpuWallEnergy64.cu"),
        "-o", str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    if result.returncode == 77:
        pytest.skip("CUDA device unavailable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout
