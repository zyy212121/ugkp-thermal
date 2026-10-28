"""True CUDA execution against the frozen CPU dry Euler controller."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT


def test_resident_material_reaction_matches_cpu(tmp_path):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file():
        pytest.skip("CUDA compiler required")
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(smi).is_file() or subprocess.run([smi, "-L"], capture_output=True).returncode:
        pytest.skip("CUDA device required")
    binary = tmp_path / "resident-material-reaction"
    compile_run = subprocess.run([nvcc, "-std=c++17", "-O2", "-Xcompiler=-fno-tree-fre", "--expt-relaxed-constexpr", "--fmad=false", "-arch=sm_89",
        "-DUGKWP_GAS_SPECIES=2", "-I"+str(SOURCE_ROOT/"applications/CHMT"),
        "-I"+str(SOURCE_ROOT/"common"), "-I"+str(VALIDATION_ROOT/"solvers/CHMT"),
        str(VALIDATION_ROOT/"solvers/CHMT/tests/resident_material_reaction_cuda.cu"),
        "-o", str(binary)], capture_output=True, text=True, timeout=240)
    (tmp_path/"build.log").write_text(compile_run.stdout+compile_run.stderr)
    assert compile_run.returncode == 0, compile_run.stdout+compile_run.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    (tmp_path/"run.log").write_text(run.stdout+run.stderr)
    print(run.stdout+run.stderr, end="")
    assert run.returncode == 0, run.stdout+run.stderr
    assert run.stdout.count("PASS resident material reaction variant=") == 8
