"""Real CUDA equivalence of resident interface and accepted SSPRK3 ledgers."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT

def test_resident_interface_matches_cpu_stage_oracle(tmp_path):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file():
        pytest.skip("CUDA compiler required")
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(smi).is_file() or subprocess.run([smi, "-L"], capture_output=True).returncode:
        pytest.skip("CUDA device required")
    app = SOURCE_ROOT / "applications/CHMT"
    fixture = VALIDATION_ROOT / "solvers/CHMT/tests/resident_interface_cuda.cu"
    binary = tmp_path / "resident-interface"
    compile_run = subprocess.run([nvcc, "-std=c++17", "-arch=sm_89", "--Ofast-compile=max",
        "-Xcompiler=-fno-tree-fre", "-DUGKWP_GAS_SPECIES=2", "-I"+str(app),
        "-I"+str(SOURCE_ROOT/"common"), "-I"+str(VALIDATION_ROOT/"solvers/CHMT"),
        str(fixture), "-o", str(binary)], capture_output=True, text=True, timeout=240)
    assert compile_run.returncode == 0, compile_run.stdout+compile_run.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout+run.stderr
    assert run.stdout.count("PASS resident interface variant=") == 4
