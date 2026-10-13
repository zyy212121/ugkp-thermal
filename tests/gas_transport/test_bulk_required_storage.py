"""Real unavailable face storage cannot masquerade as a computed zero flux."""
import subprocess
import pytest
from test_mixture_state import ROOT,HERE
from test_mixture_transport import fixture

def test_missing_legacy_owner_reports_an_error_without_status_storage(tmp_path):
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    pre=pre.replace('#define asm(...) std::abort()','#define asm(...) throw 7')
    source=tmp_path/'error.cpp';source.write_text(pre+fixture()+r'''
int main(){State s;initialise(s);s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.gasSpecies.faceStatus=nullptr;s.faceOwner[0]=s.nCells;
threadIdx.x=0;bool error=false;try{computeGasInternalFaceFluxKernel<false>(&s,.01);}catch(int){error=true;}
ck(error,"missing required owner silently published zero flux");}
''')
    exe=tmp_path/'error';p=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True);assert p.returncode==0,p.stderr
    p=subprocess.run([str(exe)],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
