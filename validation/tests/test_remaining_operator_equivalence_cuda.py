"""Refactoring oracle: production device computations retain numerical behavior."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import sys
import pytest

ROOT=SOURCE_ROOT

@pytest.mark.parametrize('app',['gasUGKP', 'FSH', 'CHT'])
def test_merged_operators_match_frozen_device_implementations(tmp_path,app):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():
        pytest.skip('CUDA compiler/device required for actual device equivalence')
    command=[sys.executable,str((VALIDATION_ROOT / 'tools/tests/fixtures/operator_unification/differential.py')),str(ROOT),str(tmp_path),app]
    result=subprocess.run(command,capture_output=True,text=True,timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS all frozen-implementation differential checks' in result.stdout
