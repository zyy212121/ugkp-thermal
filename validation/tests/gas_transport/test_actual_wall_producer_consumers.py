"""Actual BVP -> sparse exchange -> selected face and inventory SST consumers."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_sst_ale_audit import AUDIT

SETUP=r'''
#include "gasWall/WallModel.H"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
struct ActualWall {
 ugkwp::SpeciesThermoData<Real> sp[2];Real coefficients[6]={1004,0,10000,1100,0,20000};
 ugkwp::gaswall::WallInput<Real,2> input;ugkwp::gaswall::WallModelConfig<Real> config;
 ugkwp::gaswall::WallWorkspace<Real,2,24> workspace;ugkwp::gaswall::WallOutput<Real,2> output;
 ActualWall(State&s){
  for(int j=0;j<2;++j){sp[j].coefficientOffset=3*j;sp[j].molarMass=.028;sp[j].minTemperature=100;sp[j].maxTemperature=4000;}
  input.model.thermo.species=sp;input.model.thermo.coefficients=coefficients;input.model.thermo.coefficientCount=6;
  input.model.viscosity=.002;input.model.conductivity=.08;input.model.diffusivity[0]=.001;input.model.diffusivity[1]=.002;
  input.pressure=90000;input.temperature=500;input.normal[0]=1;input.velocity[1]=2;
  input.matchingDistance=.2;input.ownerDistance=.1;input.quadrature.volume=1;
  input.matching.temperature=600;input.matching.velocity[1]=20;input.matching.massFraction[0]=.7;input.matching.massFraction[1]=.3;
  input.massFlux[0]=.007;input.massFlux[1]=.003;input.matching.k=.5;input.matching.omega=200;
  config.enableSst=true;config.nodes=3;config.maxIterations=1;
  s.sstWallTreatment=2;s.sstConfigured=1;s.turbulenceModel=3;s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;
  s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.thermo=input.model.thermo;
  s.gasSpecies.rho=new Real[4]{1.4,1.4,.6,.6};s.gasSpecies.flux=new Real[6]{811,812,813,814,815,816};
  s.gasSpecies.soundSpeed=new Real[2]{400,400};s.gasSpecies.diffusivity=input.model.diffusivity;
  s.riemannBoundaryKind[1]=2;s.gasSpecies.cellStatus[0]=s.gasSpecies.faceStatus[1]=0;
  auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=new int[3]{-1,0,-1};w.ownerSlot=new int[2]{0,-1};
  w.status=new int[1]{};w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{817,818};
 }
 void produce(State&s){ugkwp::gaswall::WallStatus status;
  ck(ugkwp::gaswall::evaluateWallModel(input,config,workspace,output,status),"actual BVP unavailable");
  ck(ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(1),Real(.25)),"actual wall exchange unavailable");
 }
};
'''

@pytest.mark.parametrize('terminal',[False,True])
def test_actual_wall_selected_face_and_sst_inventory(tmp_path,terminal):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);ActualWall wall(s);
ugkp::diagnostics::DeviceContext diagnostic{};s.diagnostics=&diagnostic;wall.input.diagnostics=&diagnostic;
if(TERMINAL){wall.input.matching.temperature=std::numeric_limits<Real>::infinity();
 ugkwp::gaswall::WallStatus status;wall.output.traceTemperature=123;
 ck(!ugkwp::gaswall::evaluateWallModel(wall.input,wall.config,wall.workspace,wall.output,status),"nonfinite wall temperature accepted");
 ck(diagnostic.firstFatal.code==ugkp::diagnostics::Code::NonFinite&&std::isinf(diagnostic.firstFatal.observed),"temperature evidence lost");
 ck(diagnostic.firstFatal.sourceSiteId==2403&&diagnostic.firstFatal.component==2&&wall.output.traceTemperature==123,"invalid wall published output");return 0;}
wall.produce(s);auto&ex=s.gasBoundaryLayer.exchange[0];auto&closure=s.gasBoundaryLayer.sst[0];
ck(ex.ready&&ex.k==0,"profile completion or high-Re k failed");
ck(wall.output.traceTemperature==500&&wall.output.wallSpeciesFlux[0]==.007,"actual profile fields lost");
const Real correction=s.p[0]-wall.input.pressure;
Real expected[5]={ex.mass,ex.momentumX+correction*s.Sfx[1],ex.momentumY,ex.momentumZ,ex.energy};
// Stationary face has no pressure correction sweep; publisher's prescribed
// wall motion remains the actual physical trace/work in its exchange.
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,Real(.001));
Real got[5]={s.gasPhiRho[1],s.gasPhiRhoUx[1],s.gasPhiRhoUy[1],s.gasPhiRhoUz[1],s.gasPhiRhoE[1]};
for(int i=0;i<5;++i)ck(got[i]==expected[i]||(std::isnan(got[i])&&std::isnan(expected[i])),"selected actual bulk tail missing");
ck(s.gasSpecies.flux[1]==-.007&&s.gasSpecies.flux[4]==-.003,"selected species-major tail missing");
audit(s);threadIdx.x=0;const Real kBefore=s.rhoK[0],omegaBefore=s.rhoOmega[0];
applySstWallFunctionStateKernel(&s);
near(s.omega[0],clampMin(closure.ownerOmega,s.sstOmegaMin),"actual owner omega");
near(observed.constraintK[0],Real(0),"wall profile replaced coarse k");
near(observed.constraintOmega[0],s.rhoOmega[0]-omegaBefore,"actual omega constraint audit");
threadIdx.x=1;computeSstFaceFluxKernel(&s);
ck(s.sstPhiRhoK[1]==0&&s.sstPhiRhoOmega[1]==0,"auxiliary wall profile became inventory flux");
if(!TERMINAL){threadIdx.x=0;Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);
 const Real production=std::min(closure.wallShearStress*std::pow(s.sstWallCmu,Real(.25))*std::sqrt(s.k[0])/(s.rho[0]*s.sstWallKappa*closure.ownerDistance),s.sstCoefficients.c1*s.sstCoefficients.betaStar*s.k[0]*s.omega[0]);
 near(sk,s.rho[0]*(production-s.sstCoefficients.betaStar*s.k[0]*s.omega[0]),"actual traction/coarse-k source");
 const Real div=s.gasPhiRho[1]/sstFaceDensity(s,1)/s.V[0];
 sstSourcesForCell(s,0,div,sk,so);
 const Real dt=1e-7;applySstFluxAndSourceKernel(&s,dt);
 near(s.rhoK[0],kBefore+dt*sk,"actual SST inventory increment");
 near(observed.sourceK[0],dt*sk,"actual SST source audit");
 ck(observed.transportK[0]==0&&observed.transportOmega[0]==0,"unexpected transport audit");
 ck(std::isfinite(observed.sourceOmega[0])&&std::isfinite(observed.constraintOmega[0]),"omega audit tails missing");
 }
}
'''.replace('TERMINAL','true' if terminal else 'false'))


def test_actual_wall_positivity_energy_sink_and_outer_estimate(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
struct EnergyState:State{struct{double*gasWallEnergy;double*gasWallFlux;int*gasWallEnergyMask;}wallEnergy;};
int main(){EnergyState s;initialise(s);ActualWall wall(s);
ugkp::diagnostics::DeviceContext diagnostic{};s.diagnostics=&diagnostic;wall.input.diagnostics=&diagnostic;
wall.input.massFlux[0]=-.007;wall.input.massFlux[1]=.003;wall.produce(s);
double energy[3]{},sink[3]={901,902,903};int mask[3]={0,1,0};s.wallEnergy={energy,sink,mask};
s.gasBoundaryKind[1]=2;s.gasSpecies.positivityScale=new Real[4]{.2,.3,.4,.5};s.gasFluxPositivityScale[0]=.1;
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,Real(.001));
const Real fields[7]={s.gasPhiRho[1],s.gasPhiRhoUx[1],s.gasPhiRhoUy[1],s.gasPhiRhoUz[1],s.gasPhiRhoE[1],s.gasSpecies.flux[1],s.gasSpecies.flux[4]};
applyGasFluxPositivityScaleKernel(&s);
const Real actual[7]={s.gasPhiRho[1],s.gasPhiRhoUx[1],s.gasPhiRhoUy[1],s.gasPhiRhoUz[1],s.gasPhiRhoE[1],s.gasSpecies.flux[1],s.gasSpecies.flux[4]};
for(int j=0;j<7;++j)ck(actual[j]==fields[j]||(std::isnan(actual[j])&&std::isnan(fields[j])),"physical exchange partially limited");
ck(sink[1]==s.gasPhiRhoE[1],"limited wall energy sink differs from actual gas energy");
sink[1]=902;s.gasSpecies.faceStatus[1]=0;applyGasFluxPositivityScaleKernel(&s,true);
ck(!ugkp::diagnostics::hasFatal(&diagnostic)&&s.gasSpecies.faceStatus[1]==0&&sink[1]==s.gasPhiRhoE[1],"finite limiter audit rejected physical exchange");
ck(diagnostic.warnings[1].count>0,"limiter warning missing");
s.gasSpecies.faceStatus[1]=0;s.gasSpecies.cellStatus[0]=0;
}
''')
