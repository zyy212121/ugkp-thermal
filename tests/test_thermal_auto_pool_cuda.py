"""The same transition/consumer oracle must pass for each thermal backend."""
from pathlib import Path
import subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module',params=[('FSH',64),('CHT',64),('CHT',32)],ids=['FSH64','CHT64','CHT32'])
def consumer(request,tmp_path_factory):
    if not Path('/usr/local/cuda/bin/nvcc').is_file():pytest.skip('CUDA compiler and device required')
    app,bits=request.param
    out=tmp_path_factory.mktemp(f'thermal-auto-{app}-{bits}')
    q=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/gas_auto/gas_auto_pool.py'),str(ROOT),str(out),app,str(bits)],capture_output=True,text=True,timeout=600)
    assert q.returncode==0,q.stdout+q.stderr
    return out/'gas_auto_pool'
@pytest.mark.parametrize('block',[32,64,128])
@pytest.mark.parametrize('prebuilt',[True,False],ids=['already-L2','activate-from-L1'])
@pytest.mark.parametrize('directory',[0,1,2],ids=['full','base-no-injection','base-and-injection'])
def test_auto_prepares_and_consumes_current_directory(consumer,directory,prebuilt,block):
    q=subprocess.run([str(consumer),str(directory),str(int(prebuilt)),str(block)],capture_output=True,text=True,timeout=120)
    (consumer.parent/f'directory-{directory}-initial-{int(prebuilt)}-block-{block}.log').write_text(q.stdout+q.stderr)
    assert q.returncode==0,q.stdout+q.stderr
    assert 'PASS gas automatic scheduling followed by actual task consumers' in q.stdout
