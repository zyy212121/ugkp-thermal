"""Execute the shared inlet geometry and RNG checks on CPU and actual CUDA."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('fixture',['inlet_builder','inlet_tracking'])
def test_initial_inlet_geometry_cpu(tmp_path,fixture):
    exe=tmp_path/'geometry'
    subprocess.run(['g++','-std=c++14','-O2','-x','c++','-I'+str(ROOT/'common'),
        str(ROOT/'validation/pr41'/f'{fixture}.cu'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

@pytest.mark.parametrize('fixture',['inlet_builder','inlet_sampling','inlet_tracking'])
def test_initial_inlet_sampling_cuda(tmp_path,fixture):
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    smi=shutil.which('nvidia-smi')
    if not Path(nvcc).is_file() or not smi:
        pytest.skip('CUDA compiler and actual NVIDIA device required')
    if subprocess.run([smi,'-L'],capture_output=True).returncode:
        pytest.skip('NVIDIA device unavailable')
    exe=tmp_path/fixture
    subprocess.run([nvcc,'-std=c++14','-O2','--fmad=false','-I'+str(ROOT/'common'),
        str(ROOT/'validation/pr41'/f'{fixture}.cu'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
