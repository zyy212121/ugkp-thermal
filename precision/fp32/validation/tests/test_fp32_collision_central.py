"""Deferred preset-2 regression: these executables are not run during implementation."""
from pathlib import Path
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
ROOT=Path(SOURCE_ROOT)

def test_fp32_collision_central_host(tmp_path):
    exe=tmp_path/'central_host'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(ROOT/'validation/tests/fp32_collision_central.cpp'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

@pytest.mark.parametrize('thermal',[0,1],ids=['gas','thermal'])
def test_fp32_collision_central_cuda(tmp_path,thermal):
    nvcc=Path('/usr/local/cuda/bin/nvcc')
    if not nvcc.exists():pytest.skip('CUDA compiler is required')
    exe=tmp_path/'central_cuda'
    subprocess.run([str(nvcc),'--Ofast-compile=min','-std=c++17','-O2','--fmad=false','-arch=sm_89',
        '-DGPU_OPERATOR_THERMAL='+str(thermal),'-I'+str(ROOT/'common'),
        str(ROOT/'validation/tests/fp32_collision_central.cu'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True,timeout=300)
