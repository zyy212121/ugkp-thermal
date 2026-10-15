"""Real unavailable face storage cannot masquerade as a computed zero flux."""
import subprocess
import pytest
from test_mixture_state import ROOT,HERE,compile_probe
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


def test_skipped_moving_incidence_does_not_read_face_status(tmp_path):
    compile_probe(tmp_path,fixture()+r"""
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={2,2};
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;
s.gasGeometry.interval=.5;s.cellFaceId[1]=999;s.gasPhiRho[0]=.2;
s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;
const Real beforeE=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0,"skipped incidence changed cell status");
ck(s.rho[0]==Real(.5)-Real(.25)*Real(.2),"connected mass flux changed");
ck(s.rhoE[0]==Real(.5)*beforeE,"connected energy flux changed");
ck(s.gasSpecies.rho[0]==Real(.45)-Real(.25)*Real(.18),"connected species flux changed");}
""",sanitizers=True)
