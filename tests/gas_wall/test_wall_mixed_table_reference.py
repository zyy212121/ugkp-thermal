"""Independent long-double NASA, equilibrium, collider and Troe references."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('bits', [32, 64])
def test_borrowed_float_tables_against_independent_physical_reference(tmp_path, bits):
    source = Path(__file__).with_name('mixed_table_reference_probe.cpp')
    executable = tmp_path/f'mixed_table_{bits}'
    compiled = subprocess.run([
        'g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
        f'-DUGKWP_GPU_REAL_BITS={bits}', '-I', str(ROOT), '-I', str(ROOT/'common'),
        str(source), '-o', str(executable),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    print(run.stdout)
