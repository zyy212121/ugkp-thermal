"""Independent steady sampling keys, actual kernel replay, and restart continuity."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'validation/pr41/inlet_sequence.cu'

def test_boundary_injection_sequence_cpu(tmp_path):
    exe=tmp_path/'sequence_cpu'
    subprocess.run(['g++','-std=c++17','-O2','-x','c++','-I'+str(ROOT/'common'),str(SOURCE),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(tmp_path/'steadyState.dat')],check=True)

def test_boundary_injection_sequence_cuda(tmp_path):
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    smi=shutil.which('nvidia-smi')
    if not Path(nvcc).is_file() or not smi or subprocess.run([smi,'-L'],capture_output=True).returncode:
        pytest.skip('CUDA compiler and actual NVIDIA device required')
    exe=tmp_path/'sequence_cuda'
    subprocess.run([nvcc,'-std=c++17','-O2','--fmad=false','-I'+str(ROOT/'common'),str(SOURCE),'-o',str(exe)],check=True)
    subprocess.run([str(exe),str(tmp_path/'steadyState.dat')],check=True)
