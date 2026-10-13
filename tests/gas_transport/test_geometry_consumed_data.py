"""Operation-specific geometry liveness and actual consuming-loop errors."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

MOVING = r'''
Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;
s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;threadIdx.x=0;
'''

def test_optional_sweep_sum_does_not_veto_seven_euler_outputs(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){Real q[7]={1,2,3,4,5,.3,.7},flux[7]={1,1,1,1,1,.1,.9};
const Real sweep=Real(.75)*std::numeric_limits<Real>::max(),certificate=sweep+sweep;
ck(!std::isfinite(certificate),"overflow witness absent");
ck(ugkwp::validateGasEulerGeometry(Real(1),Real(1),certificate),"optional sweep certificate blocked geometry");
for(auto storage:{ugkwp::GasConservativeStorage::Integral,ugkwp::GasConservativeStorage::Density}){
 Real out[7]={-77,-77,-77,-77,-77,-77,-77};
 ck(ugkwp::advanceGasEulerGeometry<Real,2>(q,storage,Real(1),Real(1),certificate,Real(.1),flux,out),"optional certificate vetoed Euler");
 for(int k=0;k<7;++k)ck(out[k]==q[k]-Real(.1)*flux[k],"Euler candidate incomplete");}}
''')

def test_identity_and_integral_euler_ignore_unused_volumes(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){Real q[7]={1,2,3,4,5,.3,.7},flux[7]={1,1,1,1,1,.1,.9},out[7];
for(Real unused:{std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::quiet_NaN()}){
 for(auto storage:{ugkwp::GasConservativeStorage::Integral,ugkwp::GasConservativeStorage::Density}){
  ck(ugkwp::convertGasConservativeStorage<Real,2>(q,unused,storage,storage,out),"identity refused unused volume");
  ck(std::memcmp(q,out,sizeof q)==0,"identity changed complete candidate");}
 ck(ugkwp::advanceGasEulerGeometry<Real,2>(q,ugkwp::GasConservativeStorage::Integral,unused,unused,unused,Real(.1),flux,out),"integral Euler refused unused volumes");
 for(int k=0;k<7;++k)ck(out[k]==q[k]-Real(.1)*flux[k],"integral Euler incomplete");}
for(int mode=0;mode<4;++mode){for(auto& x:out)x=-77;
 const Real volume=mode==0?Real(0):mode==3?std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::quiet_NaN();
 ck(!ugkwp::convertGasConservativeStorage<Real,2>(q,volume,mode==2?ugkwp::GasConservativeStorage::Density:ugkwp::GasConservativeStorage::Integral,mode==2?ugkwp::GasConservativeStorage::Integral:ugkwp::GasConservativeStorage::Density,out),"required storage operation claimed success");
 for(Real x:out)ck(x==-77,"required storage failure partially published");}
}
''')

def test_pure_volume_reads_have_no_face_or_interval_dependencies(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int mode=0;mode<6;++mode){State s;initialise(s);MOVING
if(mode==0)s.gasGeometry.faceSweptVolume=nullptr;
if(mode==1)sweep[0]=std::numeric_limits<Real>::infinity();
if(mode==2)s.cellFaceId[1]=999;
if(mode==3)s.faceOwner[1]=1;
if(mode==4)s.gasGeometry.interval=std::numeric_limits<Real>::quiet_NaN();
if(mode==5){s.cellPlaneStart=nullptr;s.cellPlaneCount=nullptr;s.cellFaceId=nullptr;s.faceOwner=nullptr;s.faceNeighbour=nullptr;}
Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,std::numeric_limits<Real>::quiet_NaN(),a,b)&&a==1&&b==2,"pure volume read consumed unused metadata");}
State s;initialise(s);MOVING
for(int mode=0;mode<3;++mode){s.gasGeometry.oldVolume=mode==0?nullptr:oldV;s.gasGeometry.newVolume=mode==1?nullptr:newV;if(mode==2)oldV[0]=std::numeric_limits<Real>::quiet_NaN();Real a=-77,b=-77;ck(!ugkwp::gasGeometryCellVolumes(s,0,.5,a,b)&&a==-77&&b==-77,"required volume read failed without an error");}}
'''.replace('MOVING',MOVING))

def test_live_divergence_ignores_unused_sweeps_and_interval(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int mode=0;mode<3;++mode){State s;initialise(s);MOVING
if(mode==0)s.gasGeometry.faceSweptVolume=nullptr;
if(mode==1)sweep[0]=std::numeric_limits<Real>::infinity();
if(mode==2)s.gasGeometry.interval=std::numeric_limits<Real>::quiet_NaN();
s.gasPhiRho[0]=.2;s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;s.gasPhiRhoUx[0]=2;s.gasPhiRhoUy[0]=-3;s.gasPhiRhoUz[0]=4;s.gasPhiRhoE[0]=6;
const Real e=s.rhoE[0];applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0,"unused metadata vetoed divergence");
ck(s.rho[0]==Real(.5)-Real(.25)*Real(.2)&&s.rhoUx[0]==Real(-.5)&&s.rhoUy[0]==Real(.75)&&s.rhoUz[0]==-1&&s.rhoE[0]==Real(.5)*e-Real(1.5),"bulk candidate incomplete");
ck(s.gasSpecies.rho[0]==Real(.5)*Real(.9)-Real(.25)*Real(.18)&&s.gasSpecies.rho[2]==Real(.5)*Real(.1)-Real(.25)*Real(.02),"species candidate incomplete");}}
'''.replace('MOVING',MOVING))

def test_frame_only_consumes_frame_data_and_reports_real_division_errors(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);MOVING
s.gasGeometry.oldVolume=nullptr;s.gasGeometry.newVolume=nullptr;s.faceOwner=nullptr;sweep[0]=.25;
ugkwp::GasFaceFrame<Real> frame;frame.area=-77;
ck(ugkwp::gasGeometryFaceFrame(s,0,std::numeric_limits<Real>::quiet_NaN(),frame),"frame depended on unused cell/time data");
ck(frame.nx==1&&frame.ny==0&&frame.nz==0&&frame.area==1&&frame.meshVolumeRate==Real(.5)&&frame.wx==Real(.5)&&frame.wy==0&&frame.wz==0,"frame incomplete");
for(int mode=0;mode<4;++mode){s.gasGeometry.faceSweptVolume=mode==0?nullptr:sweep;s.gasGeometry.interval=mode==1?0:.5;sweep[0]=mode==2?std::numeric_limits<Real>::infinity():Real(.25);frame.area=-77;
ck(!ugkwp::gasGeometryFaceFrame(s,mode==3?999:0,.5,frame)&&frame.area==-77,"required frame operation claimed success");}}
'''.replace('MOVING',MOVING))

def test_live_moving_face_flux_ignores_unused_cell_geometry(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);MOVING
s.gasGeometry.oldVolume=nullptr;s.gasGeometry.newVolume=nullptr;
Real expected[7];computeGasInternalFaceFluxKernel<false>(&s,.5);
ck(s.gasSpecies.faceStatus[0]==0,"unused cell geometry blocked actual face flux");
expected[0]=s.gasPhiRho[0];expected[1]=s.gasPhiRhoUx[0];expected[2]=s.gasPhiRhoUy[0];expected[3]=s.gasPhiRhoUz[0];expected[4]=s.gasPhiRhoE[0];expected[5]=s.gasSpecies.flux[0];expected[6]=s.gasSpecies.flux[3];
s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;computeGasInternalFaceFluxKernel<false>(&s,std::numeric_limits<Real>::quiet_NaN());
Real out[7]={s.gasPhiRho[0],s.gasPhiRhoUx[0],s.gasPhiRhoUy[0],s.gasPhiRhoUz[0],s.gasPhiRhoE[0],s.gasSpecies.flux[0],s.gasSpecies.flux[3]};
ck(s.gasSpecies.faceStatus[0]==0&&std::memcmp(out,expected,sizeof out)==0,"unused dt changed complete face candidate");}
'''.replace('MOVING',MOVING))

def test_incident_address_errors_remain_in_all_live_consumers(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int operation=0;operation<6;++operation)for(int bad=0;bad<4;++bad){if(bad==1&&(operation==1||operation==6))continue;State s;initialise(s);MOVING
s.sstConfigured=1;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.rhoK[0]=Real(.1);s.rhoOmega[0]=1;s.k[0]=Real(.1);s.omega[0]=1;s.sstKMin=Real(1e-12);s.sstOmegaMin=Real(1e-12);
if(bad==0)s.cellFaceId[0]=999;if(bad==1){s.faceOwner[0]=1;s.faceNeighbour[0]=-1;}if(bad==2)s.cellPlaneStart[0]=-1;if(bad==3)s.cellPlaneCount[0]=-1;
const Real before[9]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoK[0],s.rhoOmega[0]};
switch(operation){case 0:applySstFluxAndSourceKernel(&s,.5);break;case 1:computeGasConvectiveCourantByCellKernel(&s,.5);break;case 2:computeGasDiffusionNumberKernel(&s,.5,.5);break;case 3:computeSstStabilityNumberKernel(&s,.5,.5);break;case 4:computeGasFluxPositivityScaleKernel(&s,.5);break;case 5:applyGasFluxDivergenceByCellKernel(&s,.5);break;}
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry),"required incident address error omitted");
const Real after[9]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2],s.rhoK[0],s.rhoOmega[0]};
ck(std::memcmp(before,after,sizeof before)==0,"address error partially published physical fields");}}
'''.replace('MOVING',MOVING))

def test_positivity_only_reads_its_used_old_inventory_volume(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int mode=0;mode<2;++mode){State s;initialise(s);MOVING
s.gasGeometry.newVolume=mode==0?nullptr:newV;newV[0]=std::numeric_limits<Real>::quiet_NaN();s.gasGeometry.faceSweptVolume=nullptr;s.gasGeometry.interval=std::numeric_limits<Real>::quiet_NaN();
s.gasPhiRho[0]=2;s.gasSpecies.flux[0]=Real(1.8);s.gasSpecies.flux[3]=Real(.2);
computeGasFluxPositivityScaleKernel(&s,1.);
ck(s.gasSpecies.cellStatus[0]==0,"unused new volume blocked positivity");
const Real bulk=clampRange(Real(.999)*clampMin(s.rho[0]-s.rhoMin,Real(0))/(Real(2)+GPU_OPERATOR_TINY(1.0e-300)),Real(0),Real(1));
ck(s.gasFluxPositivityScale[0]==bulk,"mass limiter formula changed");
ck(s.gasSpecies.positivityScale[0]==Real(.999)*Real(.9)/Real(1.8)&&s.gasSpecies.positivityScale[2]==Real(.999)*Real(.1)/Real(.2),"species limiter formula changed");}}
'''.replace('MOVING',MOVING))

def test_unsigned_and_disabled_consumers_do_not_read_unused_membership(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);MOVING
s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.gasGeometry.faceSweptVolume=nullptr;s.gasGeometry.interval=std::numeric_limits<Real>::quiet_NaN();s.gasPhiRhoE[0]=2;s.gasPhiRhoE[1]=6;
computeGasConvectiveCourantByCellKernel(&s,.5);ck(s.gasSpecies.cellStatus[0]==0&&s.gasFluxPositivityScale[0]==2,"unsigned Courant required membership or sweeps");
for(int kind:{3,4})for(int operation=0;operation<2;++operation){State t;initialise(t);t.sstConfigured=1;t.sstCoefficients=ugkwp::defaultSstCoefficients();t.rhoK[0]=1;t.rhoOmega[0]=1;t.k[0]=1;t.omega[0]=1;t.sstKMin=Real(1e-12);t.sstOmegaMin=Real(1e-12);t.maxDiffusionNumber=.25;
t.gasGeometry.enabled=true;t.gasGeometry.oldVolume=oldV;t.gasGeometry.newVolume=newV;t.gasGeometry.faceSweptVolume=nullptr;t.gasGeometry.interval=std::numeric_limits<Real>::quiet_NaN();t.cellPlaneStart[0]=1;t.cellPlaneCount[0]=1;t.riemannBoundaryKind[1]=kind;t.faceOwner=nullptr;t.faceNeighbour=nullptr;
threadIdx.x=0;if(operation==0)computeGasDiffusionNumberKernel(&t,0.,.5);else computeSstStabilityNumberKernel(&t,0.,.5);
ck(t.gasSpecies.cellStatus[0]==0,"disabled boundary consumed unrelated addresses");
ck(operation==0?t.gasDiffusionNumber[0]==0:t.sstSourceNumber[0]==0,"disabled boundary changed stability candidate");}}
'''.replace('MOVING',MOVING))

def test_actual_opposite_cell_access_is_reported_inline(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int operation=0;operation<3;++operation){State s;initialise(s);MOVING
s.sstConfigured=1;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.rhoK[0]=Real(.1);s.rhoOmega[0]=1;s.k[0]=Real(.1);s.omega[0]=1;s.faceNeighbour[0]=999;
if(operation==0)applySstFluxAndSourceKernel(&s,.5);else if(operation==1)computeGasDiffusionNumberKernel(&s,.5,.5);else computeSstStabilityNumberKernel(&s,.5,.5);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry),"required opposite-cell address not reported");}}
'''.replace('MOVING',MOVING))

def test_pure_volume_and_live_divergence_have_no_proof_reads(tmp_path):
    import re
    import subprocess
    from test_mixture_state import ROOT,HERE
    overlay=tmp_path/'overlay';(overlay/'gasTransport').mkdir(parents=True)
    helper=(ROOT/'common/gasTransport/GasGeometryValidation.H').read_text()
    offset=helper.index('UGKWP_GAS_HD inline bool gasGeometryCellVolumes')
    volume=helper[offset:]
    volume=re.sub(r's\.cellFaceId\[([^]]+)\]',r'(++helperIds,s.cellFaceId[\1])',volume)
    volume=re.sub(r'g\.faceSweptVolume\[([^]]+)\]',r'(++helperSweeps,g.faceSweptVolume[\1])',volume)
    (overlay/'gasTransport/GasGeometryValidation.H').write_text(helper[:offset]+volume)
    kernel=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    a=kernel.index('__global__ void applyGasFluxDivergenceByCellKernel');b=kernel.index('__global__ void blendGasConservativeStateKernel',a)
    live=re.sub(r's\.cellFaceId\[([^]]+)\]',r'(++liveIds,s.cellFaceId[\1])',kernel[a:b])
    path=tmp_path/'counted.cuh';path.write_text(kernel[:a]+live+kernel[b:])
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    pre='static int helperIds=0,helperSweeps=0,liveIds=0;\n'+pre.replace('#include "operators/computeSstFaceFluxKernel.cuh"','#include "'+str(path)+'"')
    body=r'''
int main(){State s;initialise(s);MOVING
Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,.5,a,b)&&a==1&&b==2,"volume outputs incomplete");
ck(helperIds==0&&helperSweeps==0&&liveIds==0,"pure volume used proof-only face data");
const Real e=s.rhoE[0];applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==Real(.5)&&s.rhoUx[0]==0&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==Real(.5)*e&&s.gasSpecies.rho[0]==Real(.45)&&s.gasSpecies.rho[2]==Real(.05),"live counted candidate incomplete");
ck(helperIds==0&&helperSweeps==0&&liveIds==4,"live divergence has a new proof pass or omitted its consuming loop");}
'''.replace('MOVING',MOVING)
    source=tmp_path/'probe.cpp';source.write_text(pre+fixture()+body);exe=tmp_path/'probe'
    c=subprocess.run(['g++','-std=c++17','-O0','-I'+str(overlay),'-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasTransport'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert c.returncode==0,c.stdout+c.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr

def test_empty_incidence_does_not_require_unused_offset(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(int operation=0;operation<6;++operation){State s;initialise(s);MOVING
s.cellPlaneStart[0]=-1;s.cellPlaneCount[0]=0;s.cellFaceId=nullptr;s.faceOwner=nullptr;s.faceNeighbour=nullptr;
s.sstConfigured=1;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.rhoK[0]=1;s.rhoOmega[0]=1;s.k[0]=1;s.omega[0]=1;s.sstKMin=Real(1e-12);s.sstOmegaMin=Real(1e-12);s.maxDiffusionNumber=.25;
switch(operation){case 0:applySstFluxAndSourceKernel(&s,0.);break;case 1:computeGasConvectiveCourantByCellKernel(&s,0.);break;case 2:computeGasDiffusionNumberKernel(&s,0.,.5);break;case 3:computeSstStabilityNumberKernel(&s,0.,.5);break;case 4:computeGasFluxPositivityScaleKernel(&s,0.);break;case 5:applyGasFluxDivergenceByCellKernel(&s,0.);break;}
ck(s.gasSpecies.cellStatus[0]==0,"empty loop required unused incidence offset");}}
'''.replace('MOVING',MOVING))
