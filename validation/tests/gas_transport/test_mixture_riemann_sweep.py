"""Conservative production Riemann flux vs analytic FV references.
The harness evolves complete energy and all species. It intentionally checks
convergence, not exact pressure equilibrium in mixed cells with different EOS.
"""
import csv,io,subprocess
from pathlib import Path
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.fixture(scope='module')
def sweep(tmp_path_factory):
    root=Path(SOURCE_ROOT);out=tmp_path_factory.mktemp('mixture-riemann-sweep');exe=out/'sweep'
    cmd=['g++','-std=c++17','-O2','-I'+str(root),'-I'+str(root/'common'),str(Path(__file__).parent/'fixtures/mixture_riemann_sweep.cpp'),'-o',str(exe)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return exe

@pytest.mark.parametrize('scenario,direction',[('constant-contact',-1),('constant-contact',1),('variable-contact',-1),('variable-contact',1),('sod',1),('rarefaction',1)])
def test_conservation_positivity_and_refinement(sweep,tmp_path,scenario,direction):
    measurements={}
    for scheme in [2,4,8]:
        series=[]
        for n in [64,128,256]:
            output=tmp_path/f'{scheme}-{n}';output.mkdir()
            result=subprocess.run([str(sweep),scenario,str(scheme),str(n),str(output),str(direction)],capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
            row=list(csv.DictReader(io.StringIO(result.stdout)))[0];v={k:float(x)for k,x in row.items()if k!='scenario'}
            assert v['minT']>0 and v['minRho']>0 and v['minY']>=-1e-12
            assert v['conservativeClosure']<1e-11
            series.append(v)
        for coarse,fine in zip(series,series[1:]):
            assert fine['rhoL1']<coarse['rhoL1'] and fine['pL1']<coarse['pL1']
        measurements[scheme]=series
    if 'contact'in scenario:
        assert measurements[4][-1]['rhoL1']<measurements[2][-1]['rhoL1']
        assert measurements[4][-1]['pL1']<measurements[2][-1]['pL1']
