"""Count actual molecular table reads with test-only header instrumentation."""
import subprocess
import pytest
from test_mixture_state import ROOT,HERE
from test_mixture_transport import fixture

@pytest.mark.parametrize('active',[False,True])
def test_max_diffusivity_is_lazy_and_cell_local(tmp_path,active):
    header=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    header=header.replace('s.gasSpecies.diffusivity[k]','(diffusionReads++,s.gasSpecies.diffusivity[k])')
    instrumented=tmp_path/'counted.cuh';instrumented.write_text(header)
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    pre=pre.replace('#include "operators/computeSstFaceFluxKernel.cuh"','static int diffusionReads=0;\n#include "'+str(instrumented)+'"')
    body=r'''
int main(){State s;initialise(s);s.maxDiffusionNumber=.25;Real D[2]={.01,.03};s.gasSpecies.diffusivity=D;
s.riemannBoundaryKind[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=ACTIVE;
if(!ACTIVE){s.cellPlaneStart[0]=1;s.cellPlaneCount[0]=1;}
threadIdx.x=0;computeGasDiffusionNumberKernel(&s,.01,.5);ck(diffusionReads==(ACTIVE?2:0),"molecular table reread or touched without diffusion consumer");
const Real old=s.gasDiffusionNumber[0];D[1]=.06;computeGasDiffusionNumberKernel(&s,.01,.5);
ck(diffusionReads==(ACTIVE?4:0),"local reuse persisted across calls");if(ACTIVE)ck(s.gasDiffusionNumber[0]>old,"fresh call reused stale diffusivity");}
'''.replace('ACTIVE','1' if active else '0')
    source=tmp_path/'probe.cpp';source.write_text(pre+fixture()+body)
    exe=tmp_path/'probe';r=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/operators'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([str(exe)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
