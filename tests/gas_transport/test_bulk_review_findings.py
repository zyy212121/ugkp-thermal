"""Actual production-header regressions for independent T06 findings."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_legacy_ale_transport import fixture as legacy_fixture
from test_boundary_layer_operators import SETUP
from test_sst_ale_audit import AUDIT

@pytest.mark.parametrize('unused',['0','-1','std::numeric_limits<Real>::quiet_NaN()'])
def test_moving_divergence_does_not_evaluate_unused_static_volume(tmp_path,unused):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;s.V[0]=UNUSED;s.gasPhiRhoUx[0]=2;s.gasPhiRhoE[0]=4;Real beforeE=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==0,"unused static volume blocked moving update");ck(s.rho[0]==Real(.5)&&s.rhoUx[0]==Real(-.5)&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==beforeE*Real(.5)-1,"moving bulk candidate incomplete");ck(s.gasSpecies.rho[0]==Real(.45)&&s.gasSpecies.rho[2]==Real(.05),"moving species candidate incomplete");}
'''.replace('UNUSED',unused))

def test_zero_old_volume_computes_full_moving_candidate(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={0,1},newV[2]={1,1},sweep[3]={0,0,0};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;s.gasPhiRho[0]=-2;s.gasSpecies.flux[0]=-1.8;s.gasSpecies.flux[3]=-.2;s.gasPhiRhoE[0]=-200;Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,.5,a,b)&&a==0&&b==1,"zero old volume blocked available volumes");threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==1&&s.rhoUx[0]==0&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==100,"zero old volume bulk output incomplete");ck(s.gasSpecies.rho[0]==Real(.9)&&s.gasSpecies.rho[2]==Real(.1),"zero old volume species incomplete");}
''')

def test_euler_geometry_zero_volume_identity_and_zero_interval(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){Real q[5]={1,2,3,4,5},f[5]={1,1,1,1,1},out[5]={-77,-77,-77,-77,-77};
ck(ugkwp::advanceGasEulerGeometry<Real,0>(q,ugkwp::GasConservativeStorage::Integral,Real(0),Real(0),Real(0),Real(.1),f,out),"integral Euler copy/multiply refused zero volumes");for(int k=0;k<5;++k)ck(out[k]==q[k]-Real(.1),"integral Euler output incomplete");
ck(ugkwp::advanceGasEulerGeometry<Real,0>(q,ugkwp::GasConservativeStorage::Density,Real(1),Real(1),Real(0),Real(0),f,out),"zero interval multiplication refused");for(int k=0;k<5;++k)ck(out[k]==q[k],"zero interval output incomplete");
ck(!ugkwp::advanceGasEulerGeometry<Real,0>(q,ugkwp::GasConservativeStorage::Density,Real(1),Real(0),Real(0),Real(.1),f,out),"real zero density divisor claimed complete");}
''')

@pytest.mark.parametrize('constrained',[False,True])
@pytest.mark.parametrize('audited',[False,True])
@pytest.mark.parametrize('channel',['omega','k'])
def test_moving_sst_only_required_omega_flux_can_block_k(tmp_path,constrained,audited,channel):
    compile_probe(tmp_path,legacy_fixture()+SETUP+AUDIT+r'''
int main(){State s;initialise(s);wall(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.01));
if(AUDITED)audit(s);
if(!CONSTRAINED){s.gasBoundaryLayer.enabled=false;s.riemannBoundaryKind[1]=0;}
s.CHANNEL[0]=std::numeric_limits<Real>::quiet_NaN();Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);const Real beforeK=s.rhoK[0],beforeOmega=s.rhoOmega[0];Real expected=clampMin(beforeK+Real(.01)*sk,s.rho[0]*s.sstKMin);threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);
if(CONSTRAINED&&OPTIONAL){ck(s.gasSpecies.cellStatus[0]==0,"unused omega flux blocked K");near(s.rhoK[0],expected,"complete K source missing");ck(s.rhoOmega[0]==beforeOmega,"constrained omega changed");if(AUDITED)ck(std::isnan(observed.transportOmega[0])&&std::isnan(observed.constraintOmega[0])&&std::isfinite(observed.sourceK[0]),"optional audit information fabricated or lost");}
else ck(s.gasSpecies.cellStatus[0]!=0&&s.rhoK[0]==beforeK,"required omega failure partially published K");}
'''.replace('CONSTRAINED','true' if constrained else 'false').replace('AUDITED','true' if audited else 'false').replace('CHANNEL','sstPhiRhoOmega' if channel=='omega' else 'sstPhiRhoK').replace('OPTIONAL','true' if channel=='omega' else 'false'))

def test_jayatilleke_zero_ustar_actual_heat_overflow_is_error(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){const Real big=std::numeric_limits<Real>::max();auto t=ugkpwall::sstJayatillekeThermalTransport(1,2,1,1,1,1,1,1,1,1,0,1,0,0,big);ck(!t.valid&&t.conductivity==2&&!std::isfinite(t.heatFlux),"required molecular heat overflow reported valid");}
''')

def test_optional_periodic_overflow_cannot_authorize_unbounded_repair(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
struct State4:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,4> gasSpecies;};
int main(){State backing;initialise(backing);State4 s;using Base=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>;static_cast<Base&>(s)=static_cast<Base&>(backing);
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;Real partial[8]={1,1,1,1,1,1,1,1},fl[8]={};int status[2]={};s.gasSpecies.rho=partial;s.gasSpecies.flux=fl;s.gasSpecies.faceStatus=status;s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;s.faceOwner[0]=0;s.faceOwner[1]=1;
const Real M=std::numeric_limits<Real>::max();fl[0]=-Real(.9)*M;fl[1]=0;fl[2]=-Real(.9)*M;fl[3]=0;fl[4]=M;fl[5]=M;fl[6]=M;fl[7]=M;s.gasPhiRho[0]=M;s.gasPhiRho[1]=0;
Real raw[4];for(int k=0;k<4;++k)raw[k]=Real(.5)*(fl[2*k]-fl[2*k+1]);threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
ck(status[0]==0&&status[1]==0&&s.gasPhiRho[0]==Real(.5)*M,"optional diagnostic overflow refused finite periodic candidate");for(int k=0;k<4;++k)ck(std::memcmp(&fl[2*k],&raw[k],sizeof(Real))==0&&fl[2*k+1]==-raw[k],"unbounded repair replaced finite raw periodic candidate");}
''')

def test_gks_accumulation_reports_required_output_overflow(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
#include "gasNumerics/FullSecondOrderGks.cuh"
int main(){double grad[3][5]={};ugkpgks::LocalPrimitive p{1e308,1,0,0,1e300,1};ugkpgks::SideFluxAccumulation a;ugkpgks::clearGksSideAccumulation(a);
ck(ugkpgks::accumulateGksSide(p,grad,1.4,1,a)&&std::isfinite(a.interfaceConservative[0]),"first finite GKS side not accumulated");
p.u=-1;ck(!ugkpgks::accumulateGksSide(p,grad,1.4,-1,a),"overflowed required GKS accumulation reported success");}
''')

@pytest.mark.parametrize('moving',[False,True])
@pytest.mark.parametrize('missing',['index','connection'])
def test_missing_required_incident_face_never_publishes_partial_update(tmp_path,moving,missing):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};s.gasGeometry.enabled=MOVING;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;
MISSING;s.gasPhiRho[0]=.2;s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;const Real before[7]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
const Real after[7]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
if(MOVING&&BADINDEX){
 // Retain the old failed expectation: the later moving-status pass reads the
 // invalid face index. This illegal-input case has no valid physical oracle.
 ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry),"missing incident face not reported");
 for(int k=0;k<7;++k)ck(after[k]==before[k],"missing face published partial candidate");
}else{
 // The established incidence walk skips an invalid index or an unowned face.
 // Only connected face 0 contributes: independently integrate its signed flux.
 ck(s.gasSpecies.cellStatus[0]==0,"skipped incidence blocked connected flux");
 const long double outward[7]={Real(.2),0,0,0,0,Real(.18),Real(.02)};
 for(int k=0;k<7;++k){const long double expected=(static_cast<long double>(before[k])-.5L*outward[k])/(MOVING?2.L:1.L);
  ck(std::isfinite(after[k])&&std::abs(static_cast<long double>(after[k])-expected)<=128*std::numeric_limits<Real>::epsilon()*std::max(1.L,std::abs(expected)),"connected-face physical inventory differs from independent flux integral");}
}}
'''.replace('MOVING','true' if moving else 'false').replace('BADINDEX','true' if missing=='index' else 'false').replace('MISSING','s.cellFaceId[1]=999' if missing=='index' else 's.faceOwner[1]=1;s.faceNeighbour[1]=-1'))

@pytest.mark.parametrize('volume',[-1,0])
def test_fixed_volume_sign_continues_but_required_zero_divisor_fails(tmp_path,volume):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.V[0]=VOLUME;s.gasPhiRho[0]=.2;s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;s.gasPhiRhoUx[0]=2;s.gasPhiRhoUy[0]=-3;s.gasPhiRhoUz[0]=4;s.gasPhiRhoE[0]=6;const Real e=s.rhoE[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
if(VOLUME==0){ck(s.gasSpecies.cellStatus[0]!=0&&s.rho[0]==1&&s.rhoE[0]==e,"used zero volume divisor not reported");}
else{ck(s.gasSpecies.cellStatus[0]==0,"finite negative volume refused");ck(s.rho[0]==Real(1)+Real(.5)*Real(.2)&&s.rhoUx[0]==1&&s.rhoUy[0]==Real(-1.5)&&s.rhoUz[0]==2&&s.rhoE[0]==e+3,"fixed signed bulk incomplete");ck(s.gasSpecies.rho[0]==Real(.9)+Real(.5)*Real(.18)&&s.gasSpecies.rho[2]==Real(.1)+Real(.5)*Real(.02),"fixed signed species incomplete");}}
'''.replace('VOLUME',str(volume)))

def test_molecular_heat_overflow_reaches_actual_wall_failure(tmp_path):
    from test_mixture_wall_function import wall_fixture
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){State s;wall(s);s.k[0]=0;s.gasMu=.01;s.Uy[0]=s.riemannBoundaryUy[1]=0;s.deltaCoeffs[1]=std::numeric_limits<Real>::max()/Real(250);
Real mu=-77,k=-77,q=-77;int active=0;gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active);ck(s.gasSpecies.faceStatus[1]!=0&&active==0,"actual wall consumer published infinite heat as active");
s.gasSpecies.faceStatus[1]=0;Real m=-77,x=-77,y=-77,z=-77,e=-77;ck(!computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e)&&s.gasSpecies.faceStatus[1]!=0,"full wall flux claimed complete on molecular heat overflow");}
''')

@pytest.mark.parametrize('entry',['direct','staged','finalize'])
def test_all_gks_entries_evaluate_existing_zero_time_equations(tmp_path,entry):
    call={'direct':'ugkpgks::fullSecondOrderFlux(p,p,grad,grad,1.4,1,.01,.7,0.,.01)',
          'staged':'ugkpgks::fullSecondOrderFluxStaged(p,p,grad,grad,1.4,1,.01,.7,0.,.01)',
          'finalize':'ugkpgks::finalizeGksFlux(a,1.4,1,.01,.7,0.,.01)'}[entry]
    compile_probe(tmp_path,fixture()+r'''
#include "gasNumerics/FullSecondOrderGks.cuh"
int main(){double grad[3][5]={};ugkpgks::LocalPrimitive p{1,2,3,4,5,5};ugkpgks::SideFluxAccumulation a;ugkpgks::clearGksSideAccumulation(a);ck(ugkpgks::accumulateGksSide(p,grad,1.4,1,a)&&ugkpgks::accumulateGksSide(p,grad,1.4,-1,a),"finite side accumulation failed");
auto r=CALL;ck(r.valid,"actual zero-time GKS equations refused");for(int k=0;k<5;++k)ck(r.flux[k]==0&&std::isfinite(r.molecularViscousFlux[k]),"zero-time GKS output incomplete");ck(r.molecularHeatFlux==0,"zero-time GKS heat incomplete");}
'''.replace('CALL',call))

@pytest.mark.parametrize('entry',['direct','staged','finalize'])
def test_zero_time_gks_preserves_computed_viscous_heat_fields(tmp_path,entry):
    call={'direct':'ugkpgks::fullSecondOrderFlux(p,p,grad,grad,1.4,1,.01,.7,DT,.01)',
          'staged':'ugkpgks::fullSecondOrderFluxStaged(p,p,grad,grad,1.4,1,.01,.7,DT,.01)',
          'finalize':'ugkpgks::finalizeGksFlux(a,1.4,1,.01,.7,DT,.01)'}[entry]
    compile_probe(tmp_path,fixture()+r'''
#include "gasNumerics/FullSecondOrderGks.cuh"
int main(){double grad[3][5]={{.1,.2,.3,.4,.5},{-.1,.1,.2,.3,.4},{.2,-.1,.1,.3,.4}};ugkpgks::LocalPrimitive p{1,2,3,4,5,5};ugkpgks::SideFluxAccumulation a;ugkpgks::clearGksSideAccumulation(a);ck(ugkpgks::accumulateGksSide(p,grad,1.4,1,a)&&ugkpgks::accumulateGksSide(p,grad,1.4,-1,a),"finite sides failed");auto zero=ZERO;auto ordinary=ORDINARY;ck(zero.valid&&ordinary.valid,"evaluable GKS equations incomplete");
for(int k=0;k<5;++k)ck(zero.molecularViscousFlux[k]==ordinary.molecularViscousFlux[k],"zero time skipped actual viscous operator");for(int k=0;k<4;++k)ck(zero.flux[k]==0,"zero time integrated equation differs");ck(zero.molecularHeatFlux==ordinary.molecularHeatFlux&&zero.molecularHeatFlux!=0,"zero time fabricated heat");ck(zero.flux[4]==(1./.7-1.)*zero.molecularHeatFlux,"zero time energy equation changed");}
'''.replace('ZERO',call.replace('DT','0.')).replace('ORDINARY',call.replace('DT','.01')))

def test_fixed_mixture_does_not_evaluate_unused_legacy_volume_floor(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
#include <cfenv>
int main(){State s;initialise(s);s.V[0]=-1;s.rhoMin=0;s.gasPhiRho[0]=.2;s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;feenableexcept(FE_DIVBYZERO|FE_INVALID);threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==Real(1)+Real(.5)*Real(.2),"unused legacy floor arithmetic blocked finite mixture update");}
''')

@pytest.mark.parametrize('operation',['volumes','divergence','courant','diffusion','positivity','frame'])
def test_zero_geometry_interval_only_fails_at_real_frame_division(tmp_path,operation):
    bodies={
      'volumes':r'''Real a=-77,b=-77;ck(ugkwp::gasGeometryCellVolumes(s,0,0.,a,b)&&a==1&&b==1,"unused zero interval blocked volume publication");''',
      'divergence':r'''s.gasPhiRho[0]=.2;s.gasSpecies.flux[0]=.18;s.gasSpecies.flux[3]=.02;s.gasPhiRhoE[0]=4;const Real E=s.rhoE[0];applyGasFluxDivergenceByCellKernel(&s,0.);ck(s.gasSpecies.cellStatus[0]==0&&s.rho[0]==1&&s.rhoUx[0]==0&&s.rhoUy[0]==0&&s.rhoUz[0]==0&&s.rhoE[0]==E&&s.gasSpecies.rho[0]==Real(.9)&&s.gasSpecies.rho[2]==Real(.1),"zero-time divergence identity incomplete");''',
      'courant':r'''s.gasPhiRhoE[0]=4;s.gasFluxPositivityScale[0]=-77;computeGasConvectiveCourantByCellKernel(&s,0.);ck(s.gasSpecies.cellStatus[0]==0&&s.gasFluxPositivityScale[0]==0,"zero-time Courant volume arithmetic refused");''',
      'diffusion':r'''s.maxDiffusionNumber=.25;s.gasDiffusionNumber[0]=-77;computeGasDiffusionNumberKernel(&s,0.,.5);ck(s.gasSpecies.cellStatus[0]==0&&s.gasDiffusionNumber[0]==0,"zero-time diffusion volume arithmetic refused");''',
      'positivity':r'''s.gasPhiRho[0]=2;s.gasSpecies.flux[0]=Real(1.8);s.gasSpecies.flux[3]=Real(.2);computeGasFluxPositivityScaleKernel(&s,0.);ck(s.gasSpecies.cellStatus[0]==0&&s.gasFluxPositivityScale[0]==1&&s.gasSpecies.positivityScale[0]==1&&s.gasSpecies.positivityScale[2]==1,"zero-time limiter branch changed");''',
      'frame':r'''ugkwp::GasFaceFrame<Real> frame;frame.area=-77;bool evaluated=false;
ck(!ugkwp::gasGeometryFaceFrame(s,0,0.,frame,&evaluated)&&evaluated,"zero interval did not report an evaluated invalid frame");
// A false numerical result does not imply untouched output. Check only the
// independently known finite geometric quantities, never accept NaN velocity.
ck(frame.area==1&&frame.nx==1&&frame.ny==0&&frame.nz==0,"evaluated face geometry was not published");
s.gasGeometry.faceSweptVolume=nullptr;frame.area=-77;evaluated=true;
ck(!ugkwp::gasGeometryFaceFrame(s,0,0.,frame,&evaluated)&&!evaluated&&frame.area==-77,"missing sweep storage was evaluated or published");''',

    }
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={};s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=0;threadIdx.x=0;BODY}
'''.replace('BODY',bodies[operation]))


@pytest.mark.parametrize('moving',[False,True])
@pytest.mark.parametrize('missing',['flux','faceStatus'])
def test_required_face_backing_reports_storage_error_before_publication(tmp_path,moving,missing):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={2,2},sweep[3]={0,0,0};
s.gasGeometry.enabled=MOVING;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.5;
s.gasSpecies.MISSING=nullptr;const Real before[7]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,.5);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage),"required face backing error was lost");
const Real after[7]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
for(int k=0;k<7;++k)ck(after[k]==before[k],"missing face backing published partial physical state");}
'''.replace('MOVING','true' if moving else 'false').replace('MISSING',missing))


@pytest.mark.parametrize('constrained',[False,True])
@pytest.mark.parametrize('channel',['omega','k'])
def test_strict_sst_nonfinite_face_flux_keeps_prior_inventory(tmp_path,constrained,channel):
    # Separate strict-path coverage. This does not certify the terminal NaN
    # candidate in the retained legacy-failure cases as a physical solution.
    compile_probe(tmp_path,legacy_fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.01));
if(!CONSTRAINED){s.gasBoundaryLayer.enabled=false;s.riemannBoundaryKind[1]=0;}
s.CHANNEL[0]=std::numeric_limits<Real>::quiet_NaN();const Real k=s.rhoK[0],omega=s.rhoOmega[0];
threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01,true);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::NonFiniteState),"strict bad-flux status missing");
ck(s.rhoK[0]==k&&s.rhoOmega[0]==omega,"strict bad-flux path modified physical inventory");}
'''.replace('CONSTRAINED','true' if constrained else 'false').replace('CHANNEL','sstPhiRhoOmega' if channel=='omega' else 'sstPhiRhoK'))
