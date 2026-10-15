"""Actual production CUDA automatic scheduling and cleanup regressions."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess,sys
import pytest
ROOT=SOURCE_ROOT
@pytest.mark.parametrize('app',['gasUGKP', 'FSH', 'CHT'])
def test_auto_switches_and_extracted_bodies_match_frozen_sources(tmp_path,app):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():pytest.skip('CUDA compiler and GPU required')
    q=subprocess.run([sys.executable,str((VALIDATION_ROOT / 'tools/tests/fixtures/auto_cleanup/behavior.py')),str(ROOT),str(tmp_path),app],capture_output=True,text=True,timeout=600)
    assert q.returncode==0,q.stdout+q.stderr
    assert 'PASS actual automatic low-high-low-empty' in q.stdout
