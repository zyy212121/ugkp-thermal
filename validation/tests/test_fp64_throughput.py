"""Independent root trajectories, manufactured PCR solutions and pivoted block solves."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT


def checked(command, log, timeout=120):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    log.write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_ordered_bisection_trajectory(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler required")
    binary = tmp_path / "ordered-bisection"
    checked([compiler, "-std=c++17", "-O2", "-I" + str(SOURCE_ROOT / "common"),
        str(SOURCE_ROOT / "validation/tests/ordered_bisection_probe.cpp"), "-o", str(binary)],
        tmp_path / "build.log")
    assert "PASS ordered probes" in checked([str(binary)], tmp_path / "run.log")


@pytest.mark.parametrize("unit", ["cold_wall_pcr", "wall_block_team"])
def test_fp64_device_algebra(unit, tmp_path):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(nvcc).is_file() or not Path(smi).is_file():
        pytest.skip("CUDA compiler and device required")
    if subprocess.run([smi, "-L"], capture_output=True).returncode:
        pytest.skip("CUDA device required")
    binary = tmp_path / unit
    checked([nvcc, "-std=c++17", "-O2", "--Ofast-compile=min", "--fmad=false",
        "--expt-relaxed-constexpr", "-Xcompiler=-ffp-contract=off",
        "-arch=" + os.environ.get("UGKWP_CUDA_ARCH", "sm_89"),
        "-I" + str(SOURCE_ROOT / "common"),
        str(SOURCE_ROOT / "validation/tests" / (unit + ".cu")), "-o", str(binary)],
        tmp_path / "build.log", timeout=600)
    checked([str(binary)], tmp_path / "run.log")
