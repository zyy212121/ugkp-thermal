"""Independent long-double thermal/frame audit of the native scalar arithmetic."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT


def test_formation_preserving_thermal_and_mechanical_conditioning(tmp_path):
    source = (VALIDATION_ROOT / 'tests/gas_wall/thermo_frame_probe.cpp')
    executable = tmp_path / f'thermo_frame_{64}'
    compile_result = subprocess.run([
        'g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
        '-I', str(ROOT),
        '-I', str(ROOT/'common'), str(source), '-o', str(executable),
    ], capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
