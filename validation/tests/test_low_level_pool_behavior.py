"""CPU old/new values and publication ordering; no CUDA execution."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT
@pytest.mark.parametrize('app',['gas','thermal'])
def test_pool_initialization_exact_values_and_order(app,tmp_path):
    source=((VALIDATION_ROOT / 'tests/fixtures/low_level/pool_initialization.cpp.in')).read_text()
    for key,value in {'REAL':'double','LABEL':app,'GAS':str(int(app=='gas'))}.items():source=source.replace('@'+key+'@',value)
    cpp=tmp_path/'pool.cpp';cpp.write_text(source);exe=tmp_path/'pool'
    build=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str((VALIDATION_ROOT / 'tests/fixtures/low_level')),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
