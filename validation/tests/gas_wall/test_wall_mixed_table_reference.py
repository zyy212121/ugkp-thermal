"""Independent long-double NASA, equilibrium, collider and Troe references."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT


def test_double_tables_against_independent_physical_reference(tmp_path):
    source = (VALIDATION_ROOT / 'tests/gas_wall/mixed_table_reference_probe.cpp')
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
