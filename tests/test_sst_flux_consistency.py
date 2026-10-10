"""Host-execute actual SST operators in both precisions; no CUDA claim."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]

def operators():
    p=(ROOT/'common/operators/computeGasPrimitiveGradientsKernel.cuh').read_text()
    s=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    return ('\n'+p[p.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL sstDynamicOmegaWallValue'):p.index('template<class GasState>\n__global__ void initialiseSstConservativeStateKernel')]
        +'\n'+(ROOT/'common/operators/sstVelocityInvariants.cuh').read_text()
        +'\n'+s[:s.index('template<class GasState>\n__global__ void computeGasCourantFieldKernel')]
        +'\n'+s[s.index('template<class GasState>\n__global__ void computeSstStabilityNumberKernel'):s.index('template<class GasState>\n__global__ void applyGasFluxDivergenceByCellKernel')]
        +'\n'+(ROOT/'common/operators/computeSstGradientsKernel.cuh').read_text()
        +'\n'+(ROOT/'common/operators/applyGasVolumeFractionSourceKernel.cuh').read_text().replace('asm("trap;");','std::abort();'))

@pytest.mark.parametrize('bits',[32,64])
def test_actual_sst_operators(tmp_path,bits):
    body=(ROOT/'tests/fixtures/sst_flux_consistency_host.hpp').read_text()+operators()+r'''
int fails=0;void check(const char*name,R got,R want){if(!std::isfinite(got)||std::abs(got-want)>(sizeof(R)==4?R(8e-5):R(1e-11))*std::max(R(1),std::abs(want))){std::cerr<<name<<" got="<<got<<" expected="<<want<<"\n";++fails;}}
int main(){
 DeviceState s;s.sstBoundaryKMode[0]=s.sstBoundaryOmegaMode[0]=2;s.gasPhiRho[0]=-2;
 check("inletOutlet follows negative flux despite positive owner U",sstBoundaryValue(s,0,0,false),2);
 computeSstGradientsKernel(&s);check("gradient uses inlet state",s.gradKX[0],R(.95));
 s.nut[0]=0;s.gasMu=R(.02);computeSstFaceFluxKernel(&s);
 check("inflow convection and diffusion share flux sign",s.sstPhiRhoK[0],R(-4.38));
 R cd=ugkwp::sstCrossDiffusion(s.omega[0],s.gradKX[0]*s.gradOmegaX[0],s.sstCoefficients);
 // Recompute gradients after changing molecular viscosity: F1 must follow them.
 computeSstGradientsKernel(&s);
 check("F1 is refreshed with SST gradients",s.sstF1[0],ugkwp::sstF1(s.k[0],s.omega[0],s.gasMu/s.rho[0],s.sstWallDistance[0],cd,s.sstCoefficients));
 s.gasPhiRho[0]=2;s.Ux[0]=-3;check("inletOutlet follows positive flux despite negative U",sstBoundaryValue(s,0,0,false),R(.1));
 computeSstFaceFluxKernel(&s);check("outflow suppresses inletOutlet diffusion",s.sstPhiRhoK[0],R(.2));
 s.gasPhiRho[0]=0;check("zero flux follows outlet",sstBoundaryValue(s,0,0,false),R(.1));
 s=DeviceState();s.sstWallTreatment=1;s.riemannBoundaryKind[0]=s.riemannBoundaryKind[1]=2;s.sstWallDistance[0]=R(.1);s.Ux[0]=1000;s.nut[0]=1;
 R cap=s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[0]*s.omega[0];
 check("cap final wall averaged production",sstKProductionForCell(s,0,1),cap);
 s.cellPlaneCount[0]=1;check("cap single wall production",sstKProductionForCell(s,0,1),cap);
 // Mixed viscous/log corner: average uncapped G first, then apply Pk.
 s=DeviceState();s.sstWallTreatment=1;s.riemannBoundaryKind[0]=s.riemannBoundaryKind[1]=2;s.boundaryRho[0]=R(.0001);s.boundaryRho[1]=4;s.Ux[0]=0;s.nut[0]=1;
 cap=s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[0]*s.omega[0];
 check("mixed wall average precedes final Pk cap",sstKProductionForCell(s,0,100),cap);
 // Zero strain trace but nonzero finite-volume divergence; boundary density differs from owner.
 s=DeviceState();s.gasMu=0;s.nut[0]=s.nut[1]=0;s.gasPhiRho[0]=4;s.gasPhiRho[1]=-2;
 R div=R(.25); // (4/4-2/4)/V
 R dt=R(.001),sk=s.rho[0]*(-R(2)/3*div*s.k[0]-s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
 R so=ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],div,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients);
 R oldK=s.rhoK[0],oldO=s.rhoOmega[0];computeSstStabilityNumberKernel(&s,dt,1);
 R divBound=R(.5);
 R boundK=s.rho[0]*(R(2)/3*divBound*s.k[0]+s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
 R boundO=std::max(std::abs(ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],-divBound,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients)),std::abs(ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],divBound,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients)));
 check("stability bounds face-flux compression",s.sstSourceNumber[0],std::max(std::abs(dt*boundK/oldK),std::abs(dt*boundO/oldO)));
 applySstFluxAndSourceKernel(&s,dt);check("k source uses face-flux divergence",s.rhoK[0],oldK+dt*sk);check("omega source uses face-flux divergence",s.rhoOmega[0],oldO+dt*so);
 // Internal rho interpolation uses owner-oriented face weight, also for neighbor cell.
 s=DeviceState();s.gasMu=0;s.nut[0]=s.nut[1]=0;s.nInternalFaces=1;s.faceNeighbour[0]=1;s.cellPlaneCount[0]=1;s.gasPhiRho[0]=13;
 div=1;oldK=s.rhoK[0];sk=s.rho[0]*(-R(2)/3*div*s.k[0]-s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);applySstFluxAndSourceKernel(&s,dt);check("weighted internal rho divU",s.rhoK[0],oldK+dt*sk);
 // Opposing predictor fluxes cancel; unequal positivity scaling must not
 // erase the source estimate. Bound all scale combinations at fixed state.
 s=DeviceState();s.gasMu=0;s.nut[0]=s.nut[1]=0;s.V[0]=1;s.boundaryRho[0]=s.boundaryRho[1]=1;s.gasPhiRho[0]=100;s.gasPhiRho[1]=-100;
 for(R magnitude:{R(0),R(100),R(1e15)}){
 s.gasPhiRho[0]=magnitude;s.gasPhiRho[1]=-magnitude;
 computeSstStabilityNumberKernel(&s,dt,1);R estimated=s.sstSourceNumber[0];
 R b=magnitude;
 R endpointK=s.rho[0]*(R(2)/3*b*s.k[0]+s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
 R endpointO=std::max(std::abs(ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],-b,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients)),std::abs(ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],b,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients)));
 check("affine compression envelope equals endpoint maximum",estimated,std::max(dt*endpointK/s.rhoK[0],dt*endpointO/s.rhoOmega[0]));
 for(R outgoingScale:{R(0),R(.5),R(1)})for(R incomingScale:{R(0),R(.5),R(1)}){
  R actualDiv=magnitude*outgoingScale-magnitude*incomingScale;
  R actualK=s.rho[0]*(-R(2)/3*actualDiv*s.k[0]-s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
  R actualO=ugkwp::sstOmegaSource(s.rho[0],s.k[0],s.omega[0],actualDiv,0,0,s.sstF1[0],s.sstF2[0],0,s.sstCoefficients);
  if(estimated+R(1e-6)<std::max(std::abs(dt*actualK/s.rhoK[0]),std::abs(dt*actualO/s.rhoOmega[0]))){std::cerr<<"cancelling flux underestimates positivity-scaled source\n";++fails;}
 }
 }
 // Neighbor orientation uses the identical owner-oriented density.
 s=DeviceState();s.gasMu=0;s.nut[0]=s.nut[1]=0;s.nInternalFaces=1;s.faceNeighbour[0]=1;s.cellPlaneStart[1]=0;s.cellPlaneCount[1]=1;s.gasPhiRho[0]=13;threadIdx.x=1;
 div=-1;oldK=s.rhoK[1];sk=s.rho[1]*(-R(2)/3*div*s.k[1]-s.sstCoefficients.betaStar*s.k[1]*s.omega[1]);applySstFluxAndSourceKernel(&s,dt);check("neighbor signed divergence uses same rho face",s.rhoK[1],oldK+dt*sk);threadIdx.x=0;
 s.periodic=true;s.gasPhiRho[0]=13;s.gasPhiRho[1]=-9;check("periodic predictor pair average",sstPredictorMassFlux(s,0),11);check("periodic predictor antisymmetry",sstPredictorMassFlux(s,1),-11);check("periodic predictor leaves raw flux immutable",s.gasPhiRho[0],13);
 // Physical volume source conserves primitive turbulence in unconstrained cells.
 for(R eps:{R(.1),R(.3)}){s=DeviceState();s.epsSolid[0]=s.epsSolid[1]=eps;R k=s.k[0],o=s.omega[0];applyGasVolumeFractionSourceKernel(&s,R(.01));check("epsG k primitive coherence",s.rhoK[0]/s.rho[0],k);check("epsG omega primitive coherence",s.rhoOmega[0]/s.rho[0],o);}
 s=DeviceState();s.sstConfigured=0;oldK=s.rhoK[0];oldO=s.rhoOmega[0];applyGasVolumeFractionSourceKernel(&s,R(.01));check("disabled SST k untouched",s.rhoK[0],oldK);check("disabled SST omega untouched",s.rhoOmega[0],oldO);
 return fails?1:0;
}
'''
    src=tmp_path/'probe.cpp';src.write_text(body.replace('#pragma once',''));exe=tmp_path/f'probe{bits}'
    subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(src),'-o',str(exe)],check=True)
    p=subprocess.run([str(exe)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_all_sst_gradient_consumers_have_same_stage_mass_flux():
    advance=(ROOT/'common/GpuGasAdvance.cuh').read_text().split('int blendGasRungeKuttaStage')[0]
    assert advance.index('applyGasFluxPositivityScaleKernel<<<')<advance.index('computeSstGradientsKernel<<<')<advance.index('computeSstFaceFluxKernel<<<')
    tuning=(ROOT/'common/GpuToolB1Launch.cuh').read_text()
    assert 'computeSstGradientsKernel' not in tuning.split('int launchToolB1FaceBundle')[0]
    assert tuning.index('computeGasInternalFaceFluxKernel')<tuning.index('computeSstGradientsKernel')<tuning.index('computeSstFaceFluxKernel')
    for path in ('applications/gasUGKP/private_backend/GpuResidentStrict.cu','applications/FSH/private_backend/GpuResidentStrict.cu','applications/CHT/gpu/GpuResidentStrict.cu'):
        src=(ROOT/path).read_text()
        a=src.index('computeSstGradientsKernel<<<')
        assert src.rfind('computeGasCourantFieldKernel<<<',0,a)>src.rfind('computeGasPrimitiveGradientsKernel<<<',0,a),path
    eddy=(ROOT/'common/operators/computeGasEddyViscosityKernel.cuh').read_text()
    assert 'gradKX' not in eddy and 'sstF1[c] =' not in eddy

def test_courant_scratch_no_longer_overwrites_sst_mass():
    source=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    face=source.split('template<class GasState>\n__global__ void computeGasCourantFieldKernel')[1].split('__global__ void computeGasConvectiveCourantByCellKernel')[0]
    cell=source.split('__global__ void computeGasConvectiveCourantByCellKernel')[1].split('template<class GasState>\n__global__ void computeGasDiffusionNumberKernel')[0]
    assert 'computeRiemannGasFaceFluxDevice<false, true>' in face
    assert 's.gasPhiRhoE[f] = finiteDevice(amaxSf)' in face
    assert 'sumAmaxSf += finiteOr(s.gasPhiRhoE[f]' in cell

@pytest.mark.parametrize('bits',[32,64])
def test_volume_source_validates_before_commit_and_skips_disabled_arrays(tmp_path,bits):
    pre=(ROOT/'tests/fixtures/sst_flux_consistency_host.hpp').read_text()
    pre=pre.replace('rhoK[2]={R(.2),R(1.6)},rhoOmega[2]={4,32}', 'rhoKBacking[2]={R(.2),R(1.6)},rhoOmegaBacking[2]={4,32}; R* rhoK=nullptr; R* rhoOmega=nullptr; R unused=0')
    kernel=(ROOT/'common/operators/applyGasVolumeFractionSourceKernel.cuh').read_text().replace('asm("trap;");','throw 1;')
    body=pre+kernel+r'''
int main(){
 DeviceState s;s.sstConfigured=0;applyGasVolumeFractionSourceKernel(&s,R(.01));
 if(!std::isfinite(s.rho[0])||s.rho[0]==R(2))return 1;
 s=DeviceState();s.rhoK=s.rhoKBacking;s.rhoOmega=s.rhoOmegaBacking;
 s.rhoK[0]=INFINITY;R rho=s.rho[0],mom=s.rhoUx[0],energy=s.rhoE[0],omega=s.rhoOmega[0];bool caught=false;
 try{applyGasVolumeFractionSourceKernel(&s,R(.01));}catch(int){caught=true;}
 if(!caught||s.rho[0]!=rho||s.rhoUx[0]!=mom||s.rhoE[0]!=energy||s.rhoOmega[0]!=omega)return 2;
 s.rhoK[0]=R(.2);s.rhoOmega[0]=-1;caught=false;
 try{applyGasVolumeFractionSourceKernel(&s,R(.01));}catch(int){caught=true;}
 if(!caught||s.rho[0]!=rho||s.rhoK[0]!=R(.2))return 3;
 return 0;
}
'''
    path=tmp_path/'volume.cpp';path.write_text(body.replace('#pragma once',''));exe=tmp_path/'volume'
    subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(path),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
