"""Actual SST inventory/source/flux checks using independent test-side observations."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture

AUDIT=r'''
// Test-owned observations only. No audit storage or hook is attached to State.
struct SstObservation{bool enabled=true;Real transportK[2]{},transportOmega[2]{},sourceK[2]{},sourceOmega[2]{},constraintK[2]{},constraintOmega[2]{},initialTransportK[2]{},initialTransportOmega[2]{},initialSourceK[2]{},initialSourceOmega[2]{},initialConstraintK[2]{},initialConstraintOmega[2]{},volume[2]{};} observed;
void audit(State&s){observed={};for(int c=0;c<2;++c)observed.volume[c]=s.gasGeometry.enabled?s.gasGeometry.oldVolume[c]:s.V[c];}
void samePhysical(Real got,Real expected,const char*why){if(std::isnan(expected))ck(std::isnan(got),why);else if(std::isinf(expected))ck(got==expected,why);else{ck(std::isfinite(got),why);near(got,expected,why);}}
void observedApply(State*sp,double dt,bool strict=false){auto&s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;if(strict){applySstFluxAndSourceKernel(sp,dt,true);return;}
 const Real k=s.rhoK[c],o=s.rhoOmega[c];const bool moving=s.gasGeometry.enabled;
 const Real oldV=moving?s.gasGeometry.oldVolume[c]:s.V[c],newV=moving?s.gasGeometry.newVolume[c]:s.V[c];
 Real fk=0,fo=0,div=0;bool constrained=ugkwp::gasBoundaryLayerOwnerSlot(s,c)>=0;
 for(int i=0;i<s.cellPlaneCount[c];++i){int f=s.cellFaceId[s.cellPlaneStart[c]+i];if(f<0||f>=s.nFaces)continue;Real sign=s.faceOwner[f]==c?-1:1;fk+=sign*s.sstPhiRhoK[f];fo+=sign*s.sstPhiRhoOmega[f];Real vf=s.gasPhiRho[f]/sstFaceDensity(s,f);if(moving)vf+=s.gasGeometry.faceSweptVolume[f]/dt;div-=sign*vf;
 constrained=constrained||((s.sstWallTreatment==0||s.sstWallTreatment==1||s.sstWallTreatment==2)&&f>=s.nInternalFaces&&s.riemannBoundaryKind[f]==2);}
 Real sk,so;sstSourcesForCell(s,c,div/clampMin(oldV,OfSmall),sk,so);
 const Real floorK=clampMin(s.rho[c],s.rhoMin)*s.sstKMin,floorO=clampMin(s.rho[c],s.rhoMin)*s.sstOmegaMin;
 const Real ek=moving?clampMin((k*oldV+dt*(fk+oldV*sk))/newV,floorK*oldV/newV):clampMin(finiteOr(k+dt*(fk/clampMin(s.V[c],OfSmall)+sk),floorK),floorK);
 const Real eo=moving?clampMin((o*oldV+(constrained?0:dt*(fo+oldV*so)))/newV,floorO*oldV/newV):(constrained?o:clampMin(finiteOr(o+dt*(fo/clampMin(s.V[c],OfSmall)+so),floorO),floorO));
 applySstFluxAndSourceKernel(sp,dt,false);
 samePhysical(s.rhoK[c],ek,"physical SST k differs from independent face/source prediction");samePhysical(s.rhoOmega[c],eo,"physical SST omega differs from independent face/source prediction");
 if(observed.enabled){observed.transportK[c]+=dt*fk;observed.transportOmega[c]+=dt*fo;observed.sourceK[c]+=dt*oldV*sk;observed.sourceOmega[c]+=dt*oldV*so;observed.constraintK[c]+=ek*newV-k*oldV-dt*fk-dt*oldV*sk;observed.constraintOmega[c]+=eo*newV-o*oldV-dt*fo-dt*oldV*so;observed.volume[c]=newV;}}
void observedRecover(State*sp){auto&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;Real k=s.rhoK[c],o=s.rhoOmega[c];recoverSstPrimitivesKernel(sp);if(s.sstConfigured){const Real floor=clampMin(s.rho[c],s.rhoMin)*s.sstKMin;samePhysical(s.rhoK[c],clampMin(finiteOr(k,floor),floor),"SST recovery changed physical k beyond its floor");}if(observed.enabled){observed.constraintK[c]+=(s.rhoK[c]-k)*observed.volume[c];observed.constraintOmega[c]+=(s.rhoOmega[c]-o)*observed.volume[c];}}
void observedWall(State*sp,bool strict=false){auto&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;Real k=s.rhoK[c],o=s.rhoOmega[c];applySstWallFunctionStateKernel(sp,strict);if(observed.enabled){observed.constraintK[c]+=(s.rhoK[c]-k)*observed.volume[c];observed.constraintOmega[c]+=(s.rhoOmega[c]-o)*observed.volume[c];}}
void observedSave(State*sp){saveGasConservativeStateKernel(sp);int c=blockIdx.x*blockDim.x+threadIdx.x;
observed.initialTransportK[c]=observed.transportK[c];
observed.initialTransportOmega[c]=observed.transportOmega[c];
observed.initialSourceK[c]=observed.sourceK[c];
observed.initialSourceOmega[c]=observed.sourceOmega[c];
observed.initialConstraintK[c]=observed.constraintK[c];
observed.initialConstraintOmega[c]=observed.constraintOmega[c];
}
void observedBlend(State*sp,Real a,Real b){blendGasConservativeStateKernel(sp,a,b);int c=blockIdx.x*blockDim.x+threadIdx.x;
 observed.transportK[c]=a*observed.initialTransportK[c]+b*observed.transportK[c];
observed.transportOmega[c]=a*observed.initialTransportOmega[c]+b*observed.transportOmega[c];
observed.sourceK[c]=a*observed.initialSourceK[c]+b*observed.sourceK[c];
observed.sourceOmega[c]=a*observed.initialSourceOmega[c]+b*observed.sourceOmega[c];
observed.constraintK[c]=a*observed.initialConstraintK[c]+b*observed.constraintK[c];
observed.constraintOmega[c]=a*observed.initialConstraintOmega[c]+b*observed.constraintOmega[c];
}
void checkBudget(State&s,Real k0,Real o0,int c){near(s.rhoK[c]*observed.volume[c],k0+observed.transportK[c]+observed.sourceK[c]+observed.constraintK[c],"test-side k inventory balance");near(s.rhoOmega[c]*observed.volume[c],o0+observed.transportOmega[c]+observed.sourceOmega[c]+observed.constraintOmega[c],"test-side omega inventory balance");}
#define applySstFluxAndSourceKernel observedApply
#define recoverSstPrimitivesKernel observedRecover
#define applySstWallFunctionStateKernel observedWall
#define saveGasConservativeStateKernel observedSave
#define blendGasConservativeStateKernel observedBlend
void stage(State&s,double dt){for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);recoverSstPrimitivesKernel(&s);}}
'''

@pytest.mark.parametrize('rk',[1,2,3])
def test_sst_source_constraint_transport_audit_follows_rk(tmp_path,rk):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.sstWallTreatment=0;s.gasMu=.02;s.riemannBoundaryKind[1]=2;
audit(s);Real k0[2]={s.rhoK[0],s.rhoK[1]},o0[2]={s.rhoOmega[0],s.rhoOmega[1]};
for(int c=0;c<2;++c){threadIdx.x=c;recoverSstPrimitivesKernel(&s);saveGasConservativeStateKernel(&s);}
s.sstPhiRhoK[0]=3;s.sstPhiRhoOmega[0]=-2;const double dt=.5;
stage(s,dt);
if(RK>1){stage(s,dt);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(RK==2?.5:.75),Real(RK==2?.5:.25));recoverSstPrimitivesKernel(&s);}}
if(RK==3){stage(s,dt);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/3,Real(2)/3);recoverSstPrimitivesKernel(&s);}}
for(int c=0;c<2;++c)checkBudget(s,k0[c],o0[c],c);
near(observed.transportK[0]+observed.transportK[1],0,"internal k transport did not cancel");
near(observed.transportOmega[0]+observed.transportOmega[1],0,"internal omega transport did not cancel");
ck(observed.sourceK[1]<0,"SST net source was not recorded");ck(observed.constraintOmega[0]!=0,"wall projection and suppressed omega RHS not recorded");}
'''.replace('RK',str(rk)))

def test_moving_sst_audit_closes_in_integrated_units(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);audit(s);
s.sstPhiRhoK[0]=-600;s.sstPhiRhoOmega[0]=-800;
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);checkBudget(s,6,8,c);near(observed.volume[c],newV[c],"audit volume not advanced");}
near(observed.transportK[0],Real(.6),"moving k transport was not an inventory impulse");}
''')

def test_full_moving_gas_sst_chain_closes_both_cell_inventories(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.turbulenceModel=3;
Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);audit(s);
flux(s,dt);for(int f=0;f<3;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
near(s.sstPhiRhoK[0],s.gasPhiRho[0]*3,"SST k did not use accepted ALE mass flux");near(s.sstPhiRhoOmega[0],s.gasPhiRho[0]*4,"SST omega did not use accepted ALE mass flux");
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);checkBudget(s,6,8,c);near(s.rho[c],2,"moving gas state changed");}
near(observed.transportK[0]+observed.transportK[1],0,"moving internal k flux not conservative");}
''')

def test_moving_mixture_sst_preserves_species_energy_and_audit_closure(tmp_path):
    from test_mixture_transport import fixture as mixture_fixture
    near=r'''void near(Real a,Real b,const char*m){ck(std::abs(a-b)<=Real(256)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b)),m);}'''
    compile_probe(tmp_path,mixture_fixture()+near+AUDIT+r'''
int main(){ugkwp::GasCapabilityRequest request;request.mode=ugkwp::GasMode::MixtureFrozen;request.movingGeometry=true;request.turbulenceModel=3;ck(bool(ugkwp::validateGasCapabilities(request)),"verified moving mixture SST capabilities rejected");State s;initialise(s);s.turbulenceModel=3;s.sstConfigured=1;s.sstWallTreatment=0;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.gasMu=.001;s.turbulentPrandtl=.9;s.sstKMin=1e-12;s.sstOmegaMin=1e-8;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
Real oldV[2]={1,1},newV[2]={Real(1.01),Real(.99)},sweeps[3]={Real(.01),0,0};const Real dt=Real(1e-6);
auto&g=s.gasGeometry;g.enabled=true;g.oldVolume=oldV;g.newVolume=newV;g.faceSweptVolume=sweeps;g.interval=dt;g.absoluteGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();g.relativeGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();
for(int c=0;c<2;++c){s.k[c]=.1;s.omega[c]=10;s.sstWallDistance[c]=.5;threadIdx.x=c;initialiseSstConservativeStateKernel(&s);recoverSstPrimitivesKernel(&s);}audit(s);
const Real initialE=s.rhoE[0]+s.rhoE[1];Real k0[2]={s.rhoK[0],s.rhoK[1]},o0[2]={s.rhoOmega[0],s.rhoOmega[1]};
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeSstGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,dt);ck(s.gasSpecies.faceStatus[f]==0,"moving mixture SST face rejected");}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);computeSstFaceFluxKernel(&s);}
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"moving mixture SST recovery rejected");checkBudget(s,k0[c],o0[c],c);near(s.rho[c],1,"ALE mixture density");near(s.Tgas[c],700,"ALE turbulent formation contact temperature");}
near(s.rhoE[0]*newV[0]+s.rhoE[1]*newV[1],initialE,"moving SST total gas energy");
for(int k=0;k<2;++k)near(s.gasSpecies.rho[2*k]*newV[0]+s.gasSpecies.rho[2*k+1]*newV[1],1,"moving SST species conservation");}
''')

@pytest.mark.parametrize('evaluation_volume',[1,2])
def test_mesh_motion_is_removed_from_sst_compression_and_bound(tmp_path,evaluation_volume):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;
Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);
moving(s,oldV,newV,sweeps,dt);audit(s);
s.V[0]=Real(EVALUATION_VOLUME); // Evaluation metrics need not equal stage inventories.
// Uniform stationary fluid has zero absolute div(U), despite relative ALE flux.
s.gasPhiRho[0]=-s.rho[0]*sweeps[0]/dt;
s.sstPhiRhoK[0]=s.gasPhiRho[0]*s.k[0];s.sstPhiRhoOmega[0]=s.gasPhiRho[0]*s.omega[0];
const Real sourceK=-s.rho[0]*s.sstCoefficients.betaStar*s.k[0]*s.omega[0];
threadIdx.x=0;applySstFluxAndSourceKernel(&s,dt);
near(observed.sourceK[0],dt*oldV[0]*sourceK,"mesh motion spuriously compressed SST energy");
checkBudget(s,6,8,0);
}
'''.replace('EVALUATION_VOLUME',str(evaluation_volume)))


def test_test_side_inventory_oracle_detects_changed_physical_source(tmp_path):
    """A source multiplier mutation must fail the independent inventory check."""
    from pathlib import Path
    import subprocess
    root=SOURCE_ROOT
    here=(VALIDATION_ROOT / 'tests/gas_transport')
    kernel=(root/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    expression='const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + sourceK);'
    assert expression in kernel
    mutant=tmp_path/'mutant.cuh'
    mutant.write_text(kernel.replace(expression,'const GPU_OPERATOR_REAL deltaRhoK = dt*(fluxK*invV + 2*sourceK);',1))
    prefix=(here/'gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]
    prefix=prefix.replace('#include "operators/computeSstFaceFluxKernel.cuh"', '#include "'+str(mutant)+'"')
    body=fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;audit(s);threadIdx.x=0;applySstFluxAndSourceKernel(&s,.001);}
'''
    source=tmp_path/'mutated.cpp';source.write_text(prefix+body);exe=tmp_path/'mutated'
    build=subprocess.run(['g++','-std=c++17','-O0','-I'+str(here),'-I'+str(root/'common'),'-I'+str(root/'common/operators'),'-I'+str(root/'common/gasNumerics'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'physical SST k differs from independent face/source prediction' in result.stderr
