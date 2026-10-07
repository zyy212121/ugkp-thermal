"""Actual common SST source, constraint, and RK ledger closure on host emulation."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture

AUDIT=r'''
void audit(State&s){auto&a=s.gasSstAudit;a.enabled=true;
#define ALLOCATE(name) a.name=new Real[2]{};a.initial##name=nullptr;
// The public named channels intentionally have explicit allocation coverage.
a.transportK=new Real[2]{};a.transportOmega=new Real[2]{};a.sourceK=new Real[2]{};a.sourceOmega=new Real[2]{};a.constraintK=new Real[2]{};a.constraintOmega=new Real[2]{};
a.initialTransportK=new Real[2]{};a.initialTransportOmega=new Real[2]{};a.initialSourceK=new Real[2]{};a.initialSourceOmega=new Real[2]{};a.initialConstraintK=new Real[2]{};a.initialConstraintOmega=new Real[2]{};a.volume=new Real[2]{};
for(int c=0;c<2;++c){threadIdx.x=c;prepareGasSstAuditKernel(&s);}}
void checkBudget(State&s,Real k0,Real o0,int c){auto&a=s.gasSstAudit;const Real volume=a.volume[c];
near(s.rhoK[c]*volume,k0+a.transportK[c]+a.sourceK[c]+a.constraintK[c],"SST k integrated audit does not close");
near(s.rhoOmega[c]*volume,o0+a.transportOmega[c]+a.sourceOmega[c]+a.constraintOmega[c],"SST omega integrated audit does not close");}
void stage(State&s,double dt){for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);recoverSstPrimitivesKernel(&s);}}
'''

@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('rk',[1,2,3])
def test_sst_source_constraint_transport_audit_follows_rk(tmp_path,bits,rk):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.sstWallTreatment=0;s.gasMu=.02;s.riemannBoundaryKind[1]=2;
audit(s);Real k0[2]={s.rhoK[0],s.rhoK[1]},o0[2]={s.rhoOmega[0],s.rhoOmega[1]};
for(int c=0;c<2;++c){threadIdx.x=c;recoverSstPrimitivesKernel(&s);saveGasConservativeStateKernel(&s);}
s.sstPhiRhoK[0]=3;s.sstPhiRhoOmega[0]=-2;const double dt=.5;
stage(s,dt);
if(RK>1){stage(s,dt);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(RK==2?.5:.75),Real(RK==2?.5:.25));recoverSstPrimitivesKernel(&s);}}
if(RK==3){stage(s,dt);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/3,Real(2)/3);recoverSstPrimitivesKernel(&s);}}
for(int c=0;c<2;++c)checkBudget(s,k0[c],o0[c],c);
near(s.gasSstAudit.transportK[0]+s.gasSstAudit.transportK[1],0,"internal k transport did not cancel");
near(s.gasSstAudit.transportOmega[0]+s.gasSstAudit.transportOmega[1],0,"internal omega transport did not cancel");
ck(s.gasSstAudit.sourceK[1]<0,"SST net source was not recorded");ck(s.gasSstAudit.constraintOmega[0]!=0,"wall projection and suppressed omega RHS not recorded");}
'''.replace('RK',str(rk)),bits)

@pytest.mark.parametrize('bits',[32,64])
def test_moving_sst_audit_closes_in_integrated_units(tmp_path,bits):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);audit(s);
s.sstPhiRhoK[0]=-600;s.sstPhiRhoOmega[0]=-800;
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);checkBudget(s,6,8,c);near(s.gasSstAudit.volume[c],newV[c],"audit volume not advanced");}
near(s.gasSstAudit.transportK[0],Real(.6),"moving k transport was not an inventory impulse");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_full_moving_gas_sst_chain_closes_both_cell_inventories(tmp_path,bits):
    compile_probe(tmp_path,fixture()+AUDIT+r'''
int main(){State s;initialise(s);s.sstConfigured=1;s.turbulenceModel=3;
Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=Real(.001);moving(s,oldV,newV,sweeps,dt);audit(s);
flux(s,dt);for(int f=0;f<3;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
near(s.sstPhiRhoK[0],s.gasPhiRho[0]*3,"SST k did not use accepted ALE mass flux");near(s.sstPhiRhoOmega[0],s.gasPhiRho[0]*4,"SST omega did not use accepted ALE mass flux");
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);checkBudget(s,6,8,c);near(s.rho[c],2,"moving gas state changed");}
near(s.gasSstAudit.transportK[0]+s.gasSstAudit.transportK[1],0,"moving internal k flux not conservative");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_moving_mixture_sst_preserves_species_energy_and_audit_closure(tmp_path,bits):
    from test_mixture_transport import fixture as mixture_fixture
    near=r'''void near(Real a,Real b,const char*m){ck(std::abs(a-b)<=Real(256)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b)),m);}'''
    compile_probe(tmp_path,mixture_fixture()+near+AUDIT+r'''
int main(){ugkwp::GasCapabilityRequest request;request.mode=ugkwp::GasMode::MixtureFrozen;request.movingGeometry=true;request.turbulenceModel=3;ck(bool(ugkwp::validateGasCapabilities(request)),"verified moving mixture SST capabilities rejected");State s;initialise(s);s.turbulenceModel=3;s.sstConfigured=1;s.sstWallTreatment=0;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.gasMu=.001;s.turbulentPrandtl=.9;s.sstKMin=1e-12;s.sstOmegaMin=1e-8;s.sstMaxSourceNumber=.25;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
Real oldV[2]={1,1},newV[2]={Real(1.01),Real(.99)},sweeps[3]={Real(.01),0,0};const Real dt=Real(1e-6);
auto&g=s.gasGeometry;g.enabled=true;g.oldVolume=oldV;g.newVolume=newV;g.faceSweptVolume=sweeps;g.interval=dt;g.absoluteGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();g.relativeGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();
for(int c=0;c<2;++c){s.k[c]=.1;s.omega[c]=10;s.sstWallDistance[c]=.5;threadIdx.x=c;initialiseSstConservativeStateKernel(&s);recoverSstPrimitivesKernel(&s);}audit(s);
const Real initialE=s.rhoE[0]+s.rhoE[1];Real k0[2]={s.rhoK[0],s.rhoK[1]},o0[2]={s.rhoOmega[0],s.rhoOmega[1]};
for(int c=0;c<2;++c){threadIdx.x=c;validateGasStageGeometryKernel(&s,dt);computeGasPrimitiveGradientsKernel(&s);computeSstGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,dt);ck(s.gasSpecies.faceStatus[f]==0,"moving mixture SST face rejected");}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);computeSstFaceFluxKernel(&s);}
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"moving mixture SST recovery rejected");checkBudget(s,k0[c],o0[c],c);near(s.rho[c],1,"ALE mixture density");near(s.Tgas[c],700,"ALE turbulent formation contact temperature");}
near(s.rhoE[0]*newV[0]+s.rhoE[1]*newV[1],initialE,"moving SST total gas energy");
for(int k=0;k<2;++k)near(s.gasSpecies.rho[2*k]*newV[0]+s.gasSpecies.rho[2*k+1]*newV[1],1,"moving SST species conservation");}
''',bits)
