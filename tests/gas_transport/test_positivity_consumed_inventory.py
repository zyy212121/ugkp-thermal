"""Actual limiter dispatch: inactive identity, active inventories, local reuse."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('moving',[False,True])
@pytest.mark.parametrize('mixture',[False,True])
def test_nonpositive_time_publishes_identity_without_unused_reads(tmp_path,moving,mixture):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(Real dt:{Real(0),Real(-1)}){State s;initialise(s);s.gasGeometry.enabled=MOVING;s.gasSpecies.mode=MIXTURE?ugkwp::GasMode::MixtureFrozen:ugkwp::GasMode::SingleLegacy;
s.V=nullptr;s.gasGeometry.oldVolume=nullptr;s.gasGeometry.newVolume=nullptr;s.gasGeometry.faceSweptVolume=nullptr;s.cellPlaneStart=nullptr;s.cellPlaneCount=nullptr;s.cellFaceId=nullptr;s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.gasPhiRho=nullptr;s.rho=nullptr;s.gasSpecies.rho=nullptr;s.gasSpecies.flux=nullptr;
s.gasFluxPositivityScale[0]=-77;s.gasSpecies.positivityScale[0]=s.gasSpecies.positivityScale[2]=-77;threadIdx.x=0;computeGasFluxPositivityScaleKernel(&s,dt);
ck(s.gasSpecies.cellStatus[0]==0&&s.gasFluxPositivityScale[0]==1,"inactive bulk identity missing");ck(s.gasSpecies.positivityScale[0]==(MIXTURE?Real(1):Real(-77))&&s.gasSpecies.positivityScale[2]==(MIXTURE?Real(1):Real(-77)),"inactive species identity missing or legacy touched species");}}
'''.replace('MOVING','true' if moving else 'false').replace('MIXTURE','true' if mixture else 'false'))

@pytest.mark.parametrize('moving',[False,True])
def test_positive_time_zero_outflow_does_not_read_inventory(tmp_path,moving):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int poison=0;poison<3;++poison){State s;initialise(s);s.gasGeometry.enabled=MOVING;Real invalid[2]={poison==1?std::numeric_limits<Real>::quiet_NaN():std::numeric_limits<Real>::infinity(),0};s.V=poison==0?nullptr:invalid;s.gasGeometry.oldVolume=poison==0?nullptr:invalid;s.gasGeometry.newVolume=nullptr;s.gasSpecies.rho=nullptr;s.rho=nullptr;
s.gasPhiRho[0]=-1;s.gasSpecies.flux[0]=-1;s.gasSpecies.flux[3]=-2;s.gasFluxPositivityScale[0]=-77;s.gasSpecies.positivityScale[0]=s.gasSpecies.positivityScale[2]=-77;threadIdx.x=0;computeGasFluxPositivityScaleKernel(&s,1.);
ck(s.gasSpecies.cellStatus[0]==0&&s.gasFluxPositivityScale[0]==1&&s.gasSpecies.positivityScale[0]==1&&s.gasSpecies.positivityScale[2]==1,"zero outflow read unused inventory or omitted unit factors");}}
'''.replace('MOVING','true' if moving else 'false'))

@pytest.mark.parametrize('mode',['bulk','species','both'])
def test_moving_active_limiter_selects_old_volume_and_preserves_equations(tmp_path,mode):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1};s.V=nullptr;s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=nullptr;s.gasGeometry.faceSweptVolume=nullptr;
if(!BULK)s.rho=nullptr;if(!SPECIES)s.gasSpecies.rho=nullptr;s.gasPhiRho[0]=BULK?Real(2):Real(0);s.gasSpecies.flux[0]=SPECIES?Real(1.8):Real(0);s.gasSpecies.flux[3]=SPECIES?Real(.2):Real(0);threadIdx.x=0;computeGasFluxPositivityScaleKernel(&s,1.);
ck(s.gasSpecies.cellStatus[0]==0,"moving active limiter read unused static V");const Real expected=BULK?clampRange(Real(.999)*clampMin(s.rho[0]-s.rhoMin,Real(0))/(Real(2)+GPU_OPERATOR_TINY(1.0e-300)),Real(0),Real(1)):Real(1);
ck(s.gasFluxPositivityScale[0]==expected,"active bulk arithmetic changed");ck(s.gasSpecies.positivityScale[0]==(SPECIES?Real(.999)*Real(.9)/Real(1.8):Real(1))&&s.gasSpecies.positivityScale[2]==(SPECIES?Real(.999)*Real(.1)/Real(.2):Real(1)),"active species arithmetic changed");}
'''.replace('BULK','true' if mode!='species' else 'false').replace('SPECIES','true' if mode!='bulk' else 'false'))

@pytest.mark.parametrize('moving',[False,True])
def test_active_inventory_storage_failure_is_still_an_error(tmp_path,moving):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int poison=0;poison<3;++poison)for(int speciesOnly=0;speciesOnly<2;++speciesOnly){State s;initialise(s);Real invalid[2]={poison==1?std::numeric_limits<Real>::quiet_NaN():std::numeric_limits<Real>::infinity(),0};s.gasGeometry.enabled=MOVING;s.V=poison==0?nullptr:invalid;s.gasGeometry.oldVolume=poison==0?nullptr:invalid;s.gasPhiRho[0]=speciesOnly?0:2;s.gasSpecies.flux[0]=2;threadIdx.x=0;computeGasFluxPositivityScaleKernel(&s,1.);ck(s.gasSpecies.cellStatus[0]!=0,"required inventory failure reported success");}}
'''.replace('MOVING','true' if moving else 'false'))

def test_actual_inventory_is_lazy_selected_once_and_not_persistent(tmp_path):
    import subprocess
    from test_mixture_state import ROOT,HERE
    h=(ROOT/'common/operators/computeGasFluxPositivityScaleKernel.cuh').read_text()

    h=h.replace('inventoryVolume=volumes[c];','inventoryVolume=(volumeReads++,movingReads+=ugkwp::gasMovingGeometry(s)?1:0,volumes[c]);')
    h=h.replace('s.cellFaceId[start + i]','(faceReads++,s.cellFaceId[start + i])').replace('s.cellFaceId[start+i]','(faceReads++,s.cellFaceId[start+i])')
    path=tmp_path/'counted.cuh';path.write_text(h)
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    pre='static int volumeReads=0,movingReads=0,faceReads=0;\n'+pre.replace('#include "operators/computeGasFluxPositivityScaleKernel.cuh"','#include "'+str(path)+'"')
    body=r'''
int main(){for(int moving=0;moving<2;++moving)for(int mode=0;mode<6;++mode){State s;initialise(s);Real oldV[2]={1,1};s.gasGeometry.enabled=moving;s.gasGeometry.oldVolume=oldV;
const Real dt=mode==0?Real(0):mode==1?Real(-1):Real(1);const bool bulk=mode==0||mode==1||mode==3||mode==5,species=mode==0||mode==1||mode==4||mode==5;
s.gasPhiRho[0]=bulk?Real(2):Real(0);s.gasSpecies.flux[0]=species?Real(1.8):Real(0);s.gasSpecies.flux[3]=species?Real(.2):Real(0);threadIdx.x=0;volumeReads=movingReads=faceReads=0;computeGasFluxPositivityScaleKernel(&s,dt);
const int used=mode>=3?1:0;ck(s.gasSpecies.cellStatus[0]==0&&volumeReads==used&&movingReads==(moving?used:0),"volume not lazy, selected, or once-only");ck(faceReads==(mode<2?0:6),"inactive traversal retained or active face arithmetic removed");
if(mode==4||mode==5){const Real old=s.gasSpecies.positivityScale[0];if(moving)oldV[0]=Real(.5);else s.V[0]=Real(.5);computeGasFluxPositivityScaleKernel(&s,dt);ck(volumeReads==2&&s.gasSpecies.positivityScale[0]<old,"local inventory reuse escaped its call");}
}}
'''
    source=tmp_path/'probe.cpp';source.write_text(pre+fixture()+body);exe=tmp_path/'probe'
    c=subprocess.run(['g++','-std=c++17','-O0','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert c.returncode==0,c.stdout+c.stderr
    r=subprocess.run([str(exe)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
