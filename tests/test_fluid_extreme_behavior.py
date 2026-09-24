"""Behavior tests for gas-private optimization; baseline reference stays frozen."""
from pathlib import Path
import re, subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu'
def function(source,name):
    match=re.search(r'__(?:device|global)__ (?:inline )?(?:void|PressureProjectionCell) '+name+r'\s*\(',source)
    assert match,name
    start=match.start();end=source.index('{',start)+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    prefix=re.search(r'template<bool CompactParticles = false>\s*$',source[:start])
    if prefix:start=prefix.start()
    return source[start:end]
@pytest.mark.parametrize('name',['pressure_atomic'])
def test_actual_cuda_bodies(tmp_path,name):
    text=(ROOT/'tests/fixtures/fluid_extreme'/(name+'.cpp.in')).read_text()
    source=SOURCE.read_text()
    text=re.sub(r'\{\{(\w+)\}\}',lambda m:function(source,m[1]),text)
    text=text.replace('asm("trap;");','std::abort();')
    cpp=tmp_path/'test.cpp';cpp.write_text(text);exe=tmp_path/'test'
    command=['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/gasUGKP/private_backend'),'-I'+str(ROOT/'applications/gasUGKP/gpu'),'-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
