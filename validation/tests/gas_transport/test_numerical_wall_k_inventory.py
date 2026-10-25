"""Inventory-based numerical wall-function coupling, executing common kernels."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_sst_ale_audit import AUDIT

# A deliberately incompatible auxiliary profile must never become a second
# coarse-grid k inventory. Changing the publisher/source to consume its net
# source or wall-k flux makes these regressions fail.
SETUP = r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
void wall(State&s,Real tau,Real omega){
 s.sstWallTreatment=2;s.sstConfigured=1;s.turbulenceModel=3;
 s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;
 s.riemannBoundaryKind[1]=2;
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
 w.faceSlot=new int[3]{-1,0,-1};w.ownerSlot=new int[2]{0,-1};w.status=new int[1]{};
 w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];
 w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{};
 static ugkwp::SpeciesThermoData<Real> metadata[2];static Real coefficients[6]={1004,0,0,1004,0,0};
 for(int j=0;j<2;++j){metadata[j].coefficientOffset=3*j;metadata[j].molarMass=.028;metadata[j].minTemperature=100;metadata[j].maxTemperature=4000;metadata[j].referencePressure=101325;}
 ugkwp::gaswall::WallInput<Real,2> in;in.pressure=s.p[0];in.normal[0]=-1;
 in.ownerDistance=s.sstWallDistance[0];in.quadrature.volume=s.V[0];
 in.model.thermo.species=metadata;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=6;
 ugkwp::gaswall::WallOutput<Real,2> out;out.traceTemperature=300;out.traction[1]=tau;
 out.ownerOmega=omega;out.integratedKSource=Real(-3693.492)*s.V[0];out.wallKFlux=-100;
 ck(ugkwp::publishGasBoundaryLayerOutput(s,0,in,out,s.magSf[1],Real(0)),"wall publication failed");
}
'''

def test_auxiliary_profile_cannot_drain_unrelated_owner_inventory(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);s.k[0]=1e-4;s.rhoK[0]=s.rho[0]*s.k[0];wall(s,Real(4),Real(47776.201947879716));audit(s);
threadIdx.x=0;applySstWallFunctionStateKernel(&s);
const Real old=s.rhoK[0],dt=1e-7;
threadIdx.x=1;computeSstFaceFluxKernel(&s);
ck(s.sstPhiRhoK[1]==0,"auxiliary profile wall k loss leaked into owner");
threadIdx.x=0;
const Real cap=s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[0]*s.omega[0];
const Real production=std::min(Real(4)/s.rho[0]*std::pow(s.sstWallCmu,Real(.25))*std::sqrt(s.k[0])/(s.sstWallKappa*s.sstWallDistance[0]),cap);
const Real expected=s.rho[0]*(production-s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);near(sk,expected,"production does not use numerical wall stress and actual k");
applySstFluxAndSourceKernel(&s,dt);
ck(s.gasSpecies.cellStatus[0]==0,"inventory-proportional source was rejected");
near(s.rhoK[0],old+dt*expected,"auxiliary profile source leaked into owner");
near(observed.sourceK[0],dt*expected,"applied k source ledger mismatch");
ck(std::abs(observed.constraintK[0])<Real(8)*std::numeric_limits<Real>::epsilon()*old,"source update required a k clamp");
}
''')

def test_zero_inventory_stays_zero_without_injected_tke(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);s.k[0]=s.rhoK[0]=0;s.sstKMin=0;wall(s,Real(4),Real(100));
threadIdx.x=0;applySstWallFunctionStateKernel(&s);Real sk,so;sstSourcesForCell(s,0,Real(.3),sk,so);
ck(sk==0,"zero inventory gained a wall production source");
s.gasPhiRho[1]=-.4;threadIdx.x=1;computeSstFaceFluxKernel(&s);
ck(s.sstPhiRhoK[1]==0,"blowing silently injected auxiliary k");
ck(s.sstPhiRhoOmega[1]==0,"algebraic omega also acquired an independent injected flux");
threadIdx.x=0;applySstFluxAndSourceKernel(&s,Real(.001));
ck(s.gasSpecies.cellStatus[0]==0,"zero inventory source number is undefined");ck(s.rhoK[0]==0,"zero inventory was floored or drained");
}
''')

def test_wall_stress_source_relation_and_gross_timestep(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){for(Real rho:{Real(.2),Real(2),Real(20)})for(Real k:{Real(1e-8),Real(.3),Real(30)}){
State s;initialise(s);s.rho[0]=rho;s.k[0]=k;s.rhoK[0]=rho*k;wall(s,Real(4),Real(100));threadIdx.x=0;applySstWallFunctionStateKernel(&s);
const Real p=std::min(Real(4)*std::pow(Real(.09),Real(.25))*std::sqrt(k)/(rho*Real(.41)*Real(.5)),Real(10)*Real(.09)*k*Real(100));
for(Real div:{Real(-3),Real(0),Real(3)}){Real sk,so;sstSourcesForCell(s,0,div,sk,so);near(sk,rho*(p-Real(2)/3*div*k-Real(.09)*k*100),"wall stress production units/scaling");
}
}
State s;initialise(s);const Real k=s.k[0],omega=100;
const Real tau=s.rho[0]*Real(.09)*k*omega*Real(.41)*Real(.5)/(std::pow(Real(.09),Real(.25))*std::sqrt(k));wall(s,tau,omega);threadIdx.x=0;applySstWallFunctionStateKernel(&s);
Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);near(sk,Real(0),"manufactured equilibrium source");
Real before=s.rhoK[0];applySstFluxAndSourceKernel(&s,Real(.1));ck(s.gasSpecies.cellStatus[0]==0,"finite large source activity refused");near(s.rhoK[0],before+Real(.1)*sk,"actual equilibrium source was not applied");
}
''')

def test_tangential_traction_norm_is_scale_safe(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){for(Real tau:{Real(1e200),Real(1e-200)}){
State s;initialise(s);wall(s,tau,Real(100));ck(s.gasBoundaryLayer.sst[0].wallShearStress==tau,"traction norm overflowed or underflowed");}
}
''')


def test_selected_wall_ale_source_envelope_and_rk_audit(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''

int main(){
 // Selected wall: the ALE estimator must envelope gross physical activity,
 // and mesh-volume transport must not appear as an extra compression source.
 State s;initialise(s);wall(s,Real(4),Real(100));
 Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};const Real dt=.001;
 moving(s,oldV,newV,sweeps,dt);audit(s);const Real initialOmega=s.rhoOmega[0];threadIdx.x=0;applySstWallFunctionStateKernel(&s);
 s.gasPhiRho[0]=-s.rho[0]*sweeps[0]/dt;
 s.sstPhiRhoK[0]=s.gasPhiRho[0]*s.k[0];s.sstPhiRhoOmega[0]=s.gasPhiRho[0]*s.omega[0];
 Real expected,unused;sstSourcesForCell(s,0,Real(0),expected,unused);
 const Real k0=s.rhoK[0],o0=initialOmega;applySstFluxAndSourceKernel(&s,dt);
 ck(s.gasSpecies.cellStatus[0]==0,"selected ALE source rejected legal dt");
 near(observed.sourceK[0],dt*expected,"selected wall ALE source contains spurious mesh compression");checkBudget(s,k0,o0,0);
 // Selected algebraic omega survives the same actual conservative RK blend
 // and recover path while the independent k equation remains transported.
 for(Real weight:{Real(.5),Real(.25),Real(2)/3}){
  State r;initialise(r);wall(r,Real(4),Real(100));threadIdx.x=0;applySstWallFunctionStateKernel(&r);audit(r);threadIdx.x=0;saveGasConservativeStateKernel(&r);
  const Real before=r.rhoK[0],oldOmega=r.rhoOmega[0];applySstFluxAndSourceKernel(&r,Real(.001));
  blendGasConservativeStateKernel(&r,1-weight,weight);recoverSstPrimitivesKernel(&r);
  ck(r.rhoK[0]!=before,"selected RK froze physical k");near(r.omega[0],Real(100),"selected RK lost single owner omega");checkBudget(r,before,oldOmega,0);
 }
}

''')

def test_rotated_traction_excludes_normal_component(tmp_path):
    setup=SETUP.replace('in.normal[0]=-1;', 'in.normal[0]=Real(.6);in.normal[1]=Real(.8);').replace('out.traction[1]=tau;', 'out.traction[0]=Real(.8)*tau+Real(.6)*tau/2;out.traction[1]=-Real(.6)*tau+Real(.8)*tau/2;')
    compile_probe(tmp_path,fixture()+setup+r'''int main(){for(Real tau:{Real(4),Real(1e200),Real(1e-200)}){State s;initialise(s);wall(s,tau,Real(100));ck(std::abs(s.gasBoundaryLayer.sst[0].wallShearStress/tau-1)<Real(128)*std::numeric_limits<Real>::epsilon(),"rotated tangential projection/norm scaling");}}''')
