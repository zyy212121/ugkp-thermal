"""Actual production CUDA automatic scheduling and cleanup regressions."""
from pathlib import Path
import subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app,bits',[('gasUGKP',64),('FSH',64),('CHT',64),('CHT',32)])
def test_auto_switches_and_extracted_bodies_match_frozen_sources(tmp_path,app,bits):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():pytest.skip('CUDA compiler and GPU required')
    q=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/auto_cleanup/behavior.py'),str(ROOT),str(tmp_path),app,str(bits)],capture_output=True,text=True,timeout=600)
    assert q.returncode==0,q.stdout+q.stderr
    assert 'PASS actual automatic low-high-low-empty' in q.stdout
