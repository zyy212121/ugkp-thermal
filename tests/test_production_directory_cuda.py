"""Protect production entry classification, task reuse and compaction-to-consumer order.

The fixture extracts the actual pre/compaction stages, executes real CUDA/CUB,
and observes task-builder launches through a test-only linker wrapper.
"""
from pathlib import Path
import subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module',params=[('gasUGKP',64),('FSH',64),('CHT',64),('CHT',32)],ids=['gas64','FSH64','CHT64','CHT32'])
def consumer(request,tmp_path_factory):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():pytest.skip('CUDA compiler and device required')
    app,bits=request.param;out=tmp_path_factory.mktemp(f'production-directory-{app}-{bits}')
    q=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/gas_auto/gas_auto_pool.py'),str(ROOT),str(out),app,str(bits),'production-directory'],capture_output=True,text=True,timeout=600)
    assert q.returncode==0,q.stdout+q.stderr
    return out/'gas_auto_pool'
@pytest.mark.parametrize('block',[32,128])
@pytest.mark.parametrize('scenario',[0,1,2,3],ids=['L2-skip','L2-inspect','L1-activate','L1-skip'])
@pytest.mark.parametrize('path',[0,1,2],ids=['full-fallback','no-injection','injection'])
def test_production_directory_then_auto_and_collision(consumer,path,scenario,block):
    q=subprocess.run([str(consumer),str(path),str(scenario),str(block)],capture_output=True,text=True,timeout=120)
    (consumer.parent/f'path-{path}-scenario-{scenario}-block-{block}.log').write_text(q.stdout+q.stderr)
    assert q.returncode==0,q.stdout+q.stderr
    assert 'PASS production compaction -> directory -> auto -> actual collision consumer' in q.stdout
