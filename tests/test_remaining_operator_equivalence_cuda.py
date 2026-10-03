"""Refactoring oracle: production device computations retain numerical behavior."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('app,bits',[('gasUGKP',64),('FSH',64),('CHT',64),('CHT',32)])
def test_merged_operators_match_frozen_device_implementations(tmp_path,app,bits):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():
        pytest.skip('CUDA compiler/device required for actual device equivalence')
    command=[sys.executable,str(ROOT/'tests/fixtures/operator_unification/differential.py'),str(ROOT),str(tmp_path),app,str(bits)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS all frozen-implementation differential checks' in result.stdout
