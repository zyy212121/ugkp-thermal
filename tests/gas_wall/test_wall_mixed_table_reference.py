"""Independent long-double NASA, equilibrium, collider and Troe references."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_double_tables_against_independent_physical_reference(tmp_path):
    source = Path(__file__).with_name('mixed_table_reference_probe.cpp')
    executable = tmp_path/f'mixed_table_{64}'
    compiled = subprocess.run([
        'g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
        '-I', str(ROOT), '-I', str(ROOT/'common'),
        str(source), '-o', str(executable),
    ], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    print(run.stdout)
