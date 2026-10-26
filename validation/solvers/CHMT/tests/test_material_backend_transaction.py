"""Exercise commit boundaries through the real resident material backend."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[4]
APP = ROOT / "applications/CHMT"


def checked(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_resident_material_transaction(tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("native CUDA toolkit required")
    cuda = Path(nvcc).resolve().parent.parent
    includes = ["-I" + str(APP), "-I" + str(ROOT / "common")]
    obj = tmp_path / "MaterialBackend.o"
    checked([nvcc, "-std=c++17", "-O2", "--Ofast-compile=max", "--fmad=false",
             "--expt-relaxed-constexpr", "-Xcompiler=-fno-tree-fre", "-arch=sm_89",
             *includes, "-c", APP / "gpu/MaterialBackend.cu", "-o", obj])
    exe = tmp_path / "transaction"
    checked(["g++", "-std=c++17", "-O2", "-ffp-contract=off", *includes,
             "-I" + str(cuda / "include"), Path(__file__).with_name("material_backend_transaction.cpp"),
             APP / "materials/MaterialTransport.C", obj, "-L" + str(cuda / "lib64"),
             "-Wl,-rpath," + str(cuda / "lib64"), "-lcudart", "-o", exe])
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("native CUDA device required")
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
