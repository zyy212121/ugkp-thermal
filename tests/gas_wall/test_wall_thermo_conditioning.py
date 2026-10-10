"""Independent long-double thermal/frame audit of the native scalar arithmetic."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('bits', [32, 64])
def test_formation_preserving_thermal_and_mechanical_conditioning(tmp_path, bits):
    source = Path(__file__).with_name('thermo_frame_probe.cpp')
    executable = tmp_path / f'thermo_frame_{bits}'
    compile_result = subprocess.run([
        'g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
        f'-DUGKWP_GPU_REAL_BITS={bits}', '-I', str(ROOT),
        '-I', str(ROOT/'common'), str(source), '-o', str(executable),
    ], capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
