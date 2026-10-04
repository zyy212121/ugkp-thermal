"""CPU old/new values and publication ordering; no CUDA execution."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app,real',[('gas','double'),('thermal','double'),('thermal','float')])
def test_pool_initialization_exact_values_and_order(app,real,tmp_path):
    source=(ROOT/'tests/fixtures/low_level/pool_initialization.cpp.in').read_text()
    for key,value in {'REAL':real,'LABEL':app,'GAS':str(int(app=='gas'))}.items():source=source.replace('@'+key+'@',value)
    cpp=tmp_path/'pool.cpp';cpp.write_text(source);exe=tmp_path/'pool'
    build=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT/'tests/fixtures/low_level'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
