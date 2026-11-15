"""Opt-in static material worker scheduling check; no kernel timing race."""
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

def test_material_scalar_worker_coverage(tmp_path):
    if os.environ.get("UGKP_RUN_MATERIAL_SCHEDULE_CUDA") != "1":
        pytest.skip("set UGKP_RUN_MATERIAL_SCHEDULE_CUDA=1 for the exclusive GPU checkpoint")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("CUDA toolkit unavailable")
    binary = tmp_path / "material_scalar_schedule"
    build = subprocess.run([nvcc, "-std=c++17", "-O2",
        str(SOURCE_ROOT / "validation/tests/material_scalar_schedule.cu"),
        "-o", str(binary)], capture_output=True, text=True, timeout=180)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    print(result.stdout, end="")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS") == 40
