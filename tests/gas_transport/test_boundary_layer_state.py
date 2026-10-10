"""Executable borrowed wall state contracts, including absent legacy capability."""
import pytest
from test_mixture_state import compile_probe

@pytest.mark.parametrize('bits',[32,64])
def test_borrowed_wall_stage_checks_and_legacy_identity(tmp_path,bits):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasBoundaryLayerState.H"
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{};
void ck(bool value,const char*message){if(!value){std::fprintf(stderr,"%s\n",message);std::abort();}}
int main(){LegacyGasFixture legacy{};ck(!ugkwp::gasBoundaryLayerEnabled(legacy),"legacy gained wall capability");
State s;ck(!ugkwp::gasBoundaryLayerEnabled(s),"default wall state active");s.nCells=2;s.nFaces=3;
int faces[3]={-1,0,-1},owners[2]={0,-1},status[1]={0};
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=faces;w.ownerSlot=owners;w.status=status;w.exchange=exchange;w.sst=closure;
ck(ugkwp::gasBoundaryLayerFaceSlot(s,1)==0,"face map missing");ck(ugkwp::gasBoundaryLayerFaceSlot(s,2)==-1,"nonwall face selected");ck(ugkwp::gasBoundaryLayerOwnerSlot(s,0)==0&&ugkwp::gasBoundaryLayerOwnerSlot(s,1)==-1,"owner map wrong");
ck(!ugkwp::gasBoundaryLayerSlotReady(s,0),"unevaluated closure accepted");exchange[0].ready=true;closure[0].volume=2;closure[0].ownerOmega=7;closure[0].integratedKSource=-3;
ck(ugkwp::gasBoundaryLayerSlotReady(s,0),"valid evaluated closure rejected");status[0]=1;ck(!ugkwp::gasBoundaryLayerSlotReady(s,0),"failed closure accepted");status[0]=0;
ck(!ugkwp::gasBoundaryLayerSlotReady(s,1),"out of range slot accepted");w.status=nullptr;ck(!ugkwp::gasBoundaryLayerSlotReady(s,0),"missing status accepted");
}
''',bits)

def test_boundary_layer_selector_requires_explicit_physics(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasCapabilities.H"
int main(){ugkwp::GasCapabilityRequest r;r.mode=ugkwp::GasMode::MixtureChemistry;r.turbulenceModel=3;r.sstWallTreatment=2;
if(!ugkwp::validateGasCapabilities(r))return 1;
r.sstWallTreatment=1;if(ugkwp::validateGasCapabilities(r))return 2;
r.sstWallTreatment=3;if(ugkwp::validateGasCapabilities(r))return 3;
r.sstWallTreatment=2;r.turbulenceModel=1;if(ugkwp::validateGasCapabilities(r))return 4;
r.turbulenceModel=3;r.mode=ugkwp::GasMode::SingleLegacy;if(ugkwp::validateGasCapabilities(r))return 5;
}
''')

@pytest.mark.parametrize('bits',[32,64])
def test_physical_wall_exchange_uses_total_species_and_single_enthalpy(tmp_path,bits):
    from test_mixture_transport import fixture
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
int main(){State s;initialise(s);auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
int fs[3]={-1,0,-1},os[2]={0,-1},status[1]={0};w.faceSlot=fs;w.ownerSlot=os;w.status=status;
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real speciesFlux[2];w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
ugkwp::gaswall::WallInput<Real,2> input;input.pressure=100;input.normal[0]=1;input.model.thermo=s.gasSpecies.thermo;input.quadrature.volume=3;
ugkwp::gaswall::WallOutput<Real,2> output;output.traceTemperature=700;output.traceVelocity[0]=2;output.traceVelocity[1]=3;output.wallSpeciesFlux[0]=.4;output.wallSpeciesFlux[1]=-.1;output.matchingSpeciesFlux[0]=999;output.reactionIntegral[0]=333;output.conductiveHeatFlux=50;output.traction[0]=4;output.traction[1]=5;output.wallKFlux=.6;output.integratedKSource=-7;output.ownerOmega=11;
ck(ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(2),Real(.25)),"valid wall exchange rejected");
auto near=[](Real a,Real b){return std::abs(a-b)<Real(128)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b));};
ck(near(exchange[0].mass,Real(-.6)),"mass used matching flux or chemistry");ck(near(speciesFlux[0],Real(-.8))&&near(speciesFlux[1],Real(.2)),"physical species flux sign/area");
Real h=Real(.4)*ugkwp::speciesH(0,Real(700),s.gasSpecies.thermo)-Real(.1)*ugkwp::speciesH(1,Real(700),s.gasSpecies.thermo);
ck(near(exchange[0].energy,-2*(h+Real(.3)*Real(6.5)+50+25-23)),"enthalpy, kinetic, sweep or traction work double counted");
ck(near(exchange[0].momentumX,-2*(Real(.3)*2+100-4)),"momentum wall orientation");ck(near(exchange[0].k,Real(-1.2)),"k wall flux sign");ck(closure[0].integratedKSource==-7&&closure[0].ownerOmega==11&&closure[0].volume==3,"owner closure changed units");
output.traceTemperature=std::numeric_limits<Real>::quiet_NaN();ck(!ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(2),Real(.25)),"invalid exchange accepted");ck(!exchange[0].ready,"failed output retained readiness");
}
''',bits)

@pytest.mark.parametrize('bits,nodes,tolerance,success',
    [(bits,nodes,1e-8,True) for bits in (32,64) for nodes in (0,24,48,96)]
    +[(32,48,1e-20,False),(32,96,1e-20,False)])
def test_stage_evaluation_uses_unconstrained_matching_and_preserves_inventory(tmp_path,bits,nodes,tolerance,success):
    from test_mixture_transport import fixture
    src=fixture().replace('struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};',
        '#include "gasTransport/GasBoundaryLayerModelState.H"\nstruct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;};')
    body=r'''
int main(){State s;initialise(s);s.gasMu=.02;s.gasThermalConductivity=.5;s.Uy[1]=10;
s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=500;
Real diffusion[2]={.01,.01};s.gasSpecies.diffusivity=diffusion;
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;int fs[3]={-1,0,-1},os[2]={0,-1},status[1]={0};w.faceSlot=fs;w.ownerSlot=os;w.status=status;
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real speciesFlux[2];w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
auto&m=s.gasBoundaryLayerModel;m.config.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
ugkwp::gaswall::WallInput<Real,2> in[1];ugkwp::gaswall::WallOutput<Real,2> out[1];ugkwp::gaswall::WallStatus modelStatus[1];auto*workspace=new ugkwp::gaswall::WallWorkspace<Real,2>[1];
int face[1]={1},offsets[2]={0,1},matching[1]={1};Real weights[1]={1},qd[1]={.5},qw[1]={1};m.input=in;m.output=out;m.workspace=nullptr;m.workspaceCount=0;delete[] workspace;m.status=modelStatus;m.faces=face;m.matchingOffsets=offsets;m.matchingCells=matching;m.matchingWeights=weights;
in[0].normal[0]=1;in[0].matchingDistance=1.5;in[0].ownerDistance=.5;in[0].quadrature={qd,qw,1,1,.5};
// Real device scratch is raw allocation, never constructor-zeroed.
if(m.workspace)for(std::size_t b=0;b<ugkwp::gasBoundaryLayerWorkspaceBytes<Real,2>(m.config.nodes);++b)static_cast<unsigned char*>(m.workspace)[b]=0xff;
Real savedRho=s.rho[0],savedEnergy=s.rhoE[0],savedSpecies=s.gasSpecies.rho[0];
if(!ugkwp::evaluateGasBoundaryLayerSlot(s,0)){std::fprintf(stderr,"wall failed transport=%d code=%d iterations=%d residual=%.17g\n",status[0],int(modelStatus[0].code),modelStatus[0].iteration,modelStatus[0].residual);return 1;}ck(exchange[0].ready,"stage not published");
ck(in[0].matching.temperature==s.Tgas[1]&&in[0].matching.velocity[1]==10,"matching cell not refreshed");
ck(s.rho[0]==savedRho&&s.rhoE[0]==savedEnergy&&s.gasSpecies.rho[0]==savedSpecies,"auxiliary solve changed bulk inventory");
os[1]=0;ck(!ugkwp::evaluateGasBoundaryLayerSlot(s,0),"circular constrained matching donor accepted");ck(s.gasSpecies.faceStatus[1]!=0&&!exchange[0].ready,"failed stage not invalidated");
}
'''
    if nodes:
        capacity=24 if nodes<=24 else 48 if nodes<=48 else 128
        body=body.replace('m.config.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;',f'm.config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;m.config.nodes={nodes};m.config.relativeTolerance=Real({tolerance});')
        body=body.replace('m.workspace=nullptr;m.workspaceCount=0;delete[] workspace;',f'delete[] workspace;m.workspace=new ugkwp::gaswall::WallWorkspace<Real,2,{capacity}>[1];m.workspaceCount=1;m.workspaceCapacity={capacity};')
    if not success:
        # Requested precision remains authoritative: no silent relaxation or
        # publication of a failed fixed-FP64 profile below its attainable floor.
        body=body.replace("m.config.relativeTolerance=Real(1e-20);", "m.config.relativeTolerance=Real(1e-20);m.config.absoluteTolerance=0;")
        start=body.index('if(!ugkwp::evaluateGasBoundaryLayerSlot(s,0))')
        end=body.index('ck(exchange[0].ready',start)
        body=body[:start]+r'''ck(!ugkwp::evaluateGasBoundaryLayerSlot(s,0),"unattainable requested tolerance unexpectedly accepted");
ck(status[0]==int(ugkwp::GasTransportCode::BoundaryLayerFailure)&&modelStatus[0].code==ugkwp::gaswall::WallCode::NonConvergence,"wrong terminal failure category");
ck(modelStatus[0].residual>double(m.config.relativeTolerance)&&m.config.relativeTolerance==Real(1e-20),"requested tolerance silently relaxed");
ck(!exchange[0].ready&&s.rho[0]==savedRho&&s.rhoE[0]==savedEnergy&&s.gasSpecies.rho[0]==savedSpecies,"failed profile changed inventory or readiness");return 0;
'''+body[end:]
    compile_probe(tmp_path,src+body,bits)
