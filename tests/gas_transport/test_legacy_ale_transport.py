"""Execute production five-field ALE/SST kernels; no species allocations or CUDA claims."""
from pathlib import Path
import re
import pytest
from test_mixture_state import compile_probe

HERE = Path(__file__).resolve().parent


def fixture():
    fields = re.findall(r'    ([\w:]+\*) (\w+)\{\};', (HERE/'legacy_gas_fixture.hpp').read_text())
    allocate = '\n'.join(f's.{name}=new {ty[:-1]}[8]{{}};' for ty, name in fields)
    return r'''
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
void ck(bool a,const char*m){if(!a){std::fprintf(stderr,"%s\n",m);std::abort();}}
void near(Real a,Real b,const char*m){if(std::abs(a-b)>Real(128)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b))){std::fprintf(stderr,"%s: %.17g != %.17g\n",m,double(a),double(b));std::abort();}}
void initialise(State&s){
ALLOCATE
s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gasFluxScheme=1;s.gasReconstruction=0;s.gasLimiter=0;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.gasMu=0;s.gasPrClamped=.7;s.rhoMin=1e-12;s.TgasMin=1e-9;s.maxDiffusionNumber=.25;
s.gasSpecies.cellStatus=new int[2]{};s.gasSpecies.faceStatus=new int[3]{};
s.sstCoefficients=ugkwp::defaultSstCoefficients();s.sstKMin=1e-12;s.sstOmegaMin=1e-9;s.sstMaxSourceNumber=.25;s.turbulentPrandtl=.9;
for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.cellLength[c]=1;s.rho[c]=2;s.p[c]=100000;s.Tgas[c]=s.p[c]/(s.rho[c]*s.Rgas);s.rhoE[c]=s.p[c]/(s.gammaGas-1);s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;s.k[c]=3;s.omega[c]=4;s.rhoK[c]=6;s.rhoOmega[c]=8;s.sstWallDistance[c]=.5;}
s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;s.gasHllcAdcSensor[f]=1;}
blockDim.x=1;blockIdx.x=threadIdx.x=0;
}
void moving(State&s,Real*oldV,Real*newV,Real*sweeps,Real dt){s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweeps;s.gasGeometry.interval=dt;s.gasGeometry.absoluteGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();s.gasGeometry.relativeGeometryTolerance=Real(16)*std::numeric_limits<Real>::epsilon();}
void flux(State&s,double dt){for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,dt);}for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);}}
'''.replace('ALLOCATE', allocate)


@pytest.mark.parametrize('bits', [32, 64])
def test_legacy_all_schemes_moving_uniform_state_without_species(tmp_path, bits):
    compile_probe(tmp_path, fixture()+r'''
int main(){for(int scheme=1;scheme<=9;++scheme){State s;initialise(s);s.gasFluxScheme=scheme;Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);flux(s,dt);
near(s.gasPhiRho[0],-s.rho[0]*sweeps[0]/dt,"legacy ALE mass flux");near(s.gasPhiRhoE[0],-s.rhoE[0]*sweeps[0]/dt,"legacy ALE energy flux");
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"legacy ALE status");near(s.rho[c],Real(2),"moving free-stream density");near(s.p[c],Real(100000),"moving free-stream pressure");near(s.rhoUx[c],Real(0),"moving free-stream momentum");}
ck(!s.gasSpecies.rho&&!s.gasSpecies.flux&&!s.gasSpecies.soundSpeed,"legacy acquired species storage");}}
''', bits)


@pytest.mark.parametrize('bits', [32, 64])
def test_moving_sst_inventory_uses_old_and_new_volumes(tmp_path, bits):
    compile_probe(tmp_path, fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);a.sstConfigured=b.sstConfigured=1;
Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(b,oldV,newV,sweeps,dt);
a.sstPhiRhoK[0]=b.sstPhiRhoK[0]=Real(-600);a.sstPhiRhoOmega[0]=b.sstPhiRhoOmega[0]=Real(-800);
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&a,dt);applySstFluxAndSourceKernel(&b,dt);near(b.rhoK[c]*newV[c],a.rhoK[c]*oldV[c],"SST k inventory diluted incorrectly");near(b.rhoOmega[c]*newV[c],a.rhoOmega[c]*oldV[c],"SST omega inventory diluted incorrectly");near(b.sstSourceNumber[c],a.sstSourceNumber[c],"SST physical source stability changed with geometric dilution");}}
''', bits)


@pytest.mark.parametrize('bits', [32, 64])
def test_legacy_invalid_geometry_and_flux_fail_transactionally(tmp_path, bits):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);
Real before[5]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]};
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.002));ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry),"divergence accepted stale stage interval");
s.gasSpecies.cellStatus[0]=0;s.gasPhiRhoE[0]=std::numeric_limits<Real>::quiet_NaN();applyGasFluxDivergenceByCellKernel(&s,dt);ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::NonFiniteState),"legacy nonfinite flux not rejected");
Real after[5]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0]};ck(std::memcmp(before,after,sizeof before)==0,"rejected moving update published state");}
''', bits)


@pytest.mark.parametrize('bits', [32, 64])
def test_legacy_moving_cfl_guards_missing_geometry_and_uses_smaller_volume(tmp_path, bits):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);
for(int f=0;f<3;++f){threadIdx.x=f;computeGasCourantFieldKernel(&s,dt);}
threadIdx.x=1;const Real expected=Real(.5)*dt*(s.gasPhiRho[0]+s.gasPhiRho[2])/newV[1];computeGasConvectiveCourantByCellKernel(&s,dt);near(s.gasFluxPositivityScale[1],expected,"moving Courant used evaluation volume rather than minimum stage volume");
s.gasGeometry.faceSweptVolume=nullptr;threadIdx.x=0;computeGasCourantFieldKernel(&s,dt);ck(s.gasSpecies.faceStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry),"missing geometry was not a status failure");ck(s.gasPhiRho[0]==OfGreat,"invalid geometry did not force CFL rejection");}
''', bits)


def test_transaction_preflight_applies_to_null_species_legacy_host(tmp_path):
    import subprocess
    from test_gas_advance_view import PREAMBLE
    root=HERE.parents[1]
    source=re.sub(r'<<<[\s\S]*?>>>','',(root/'common/GpuGasAdvance.cuh').read_text())
    (tmp_path/'protocol.hpp').write_text(source)
    extras=r'''
#include "gasTransport/GasStateView.H"
struct SpeciesTag {static constexpr int speciesCount=2;ugkwp::GasMode mode=ugkwp::GasMode::SingleLegacy;};
struct LegacyHost:IndependentHost{SpeciesTag gasSpecies;double*V=nullptr;};
template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*){trace.push_back("unexpected-chemistry");}
template<class S>void computeGasCourantFieldKernel(S*,double){trace.push_back("wave-speed");}
template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){trace.push_back("courant");}
template<class S>void computeGasDiffusionNumberKernel(S*,double,double){trace.push_back("diffusion-bound");}
struct Policy {
 static const double*stageVolumes(LegacyHost*h,bool){return h->V;}
 static int captureChemistryAudit(LegacyHost*,bool){return 1;}
 static int validate(LegacyHost*){return 0;}
 static double targetMaxCo(LegacyHost*){return .5;}
 static int validateTimeStep(LegacyHost*,double){trace.push_back("check-CFL");return 1;}
};
'''
    main=r'''
int main(){LegacyHost h;Payload p;h.deviceState=&p;
if(prepareGasTrialTransport<Policy>(&h,.1)!=1)return 1;
if(trace.back()!="check-CFL")return 2;
for(auto&s:trace)if(s=="unexpected-chemistry")return 3;}
'''
    cpp=tmp_path/'legacy_host.cpp';cpp.write_text(PREAMBLE+extras+'\n#include "protocol.hpp"\n'+main)
    exe=tmp_path/'legacy_host';result=subprocess.run(['g++','-std=c++17','-I'+str(root/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.mark.parametrize('bits',[32,64])
def test_moving_periodic_pair_rejects_mismatched_sweeps_before_averaging(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.nFaces=2;s.nInternalFaces=0;s.faceOwner[0]=0;s.faceOwner[1]=1;s.faceNeighbour[0]=1;s.faceNeighbour[1]=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;s.Sfx[0]=1;s.Sfx[1]=-1;
Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[2]={Real(.1),Real(-.12)};moving(s,oldV,newV,sweeps,Real(.001));s.gasPhiRho[0]=2;s.gasPhiRho[1]=-3;
threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);ck(s.gasSpecies.faceStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry)&&s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidGeometry),"mismatched periodic sweeps accepted");ck(s.gasPhiRho[0]==2&&s.gasPhiRho[1]==-3,"invalid periodic geometry averaged fluxes");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_sst_inlet_outlet_uses_mesh_relative_direction(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.riemannBoundaryKind[2]=0;s.sstBoundaryKMode[2]=2;s.sstBoundaryK[2]=7;s.Ux[1]=1;
Real oldV[2]={1,1},newV[2]={1,Real(1.002)},sweeps[3]={0,0,Real(.002)};moving(s,oldV,newV,sweeps,Real(.001));near(sstBoundaryValue(s,2,1,false),7,"SST inletOutlet used inertial outflow instead of relative inflow");}
''',bits)
