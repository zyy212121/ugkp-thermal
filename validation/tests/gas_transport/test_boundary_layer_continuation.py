"""Publish complete real wall exchange from every actual matching donor."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

BODY=r'''
int main(){State s;initialise(s);s.nCells=34;s.gasMu=.02;s.gasThermalConductivity=.5;
Real rho[34],p[34],T[34],ux[34],uy[34],uz[34],species[68];int owners[34];
for(int i=0;i<34;++i){rho[i]=1;p[i]=100000+i;T[i]=700+i;ux[i]=0;uy[i]=10;uz[i]=0;owners[i]=-1;species[i]=.2;species[34+i]=.8;}
s.rho=rho;s.p=p;s.Tgas=T;s.Ux=ux;s.Uy=uy;s.Uz=uz;s.gasSpecies.rho=species;
s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=500;
Real diffusion[2]={.01,.01};s.gasSpecies.diffusivity=diffusion;
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;int fs[3]={-1,0,-1},status[1]={-123};
owners[0]=0;w.faceSlot=fs;w.ownerSlot=owners;w.status=status;
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];
Real speciesFlux[2]={123,123};w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
auto&m=s.gasBoundaryLayerModel;m.config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
m.config.nodes=3;m.config.maxIterations=1;m.config.absoluteTolerance=0;m.config.relativeTolerance=Real(1e-30);
ugkwp::gaswall::WallInput<Real,2> in[1];ugkwp::gaswall::WallOutput<Real,2> out[1];ugkwp::gaswall::WallStatus modelStatus[1];
auto*workspace=new ugkwp::gaswall::WallWorkspace<Real,2,24>[1];
int face[1]={1},offsets[2]={0,33},matching[33];Real weights[33];Real expectedT=0,expectedP=0;
for(int i=0;i<33;++i){matching[i]=i+1;weights[i]=Real(1)/33;expectedT+=weights[i]*T[i+1];expectedP+=weights[i]*p[i+1];}
m.input=in;m.output=out;m.workspace=workspace;m.workspaceCount=1;m.workspaceCapacity=24;m.status=modelStatus;
m.faces=face;m.matchingOffsets=offsets;m.matchingCells=matching;m.matchingWeights=weights;
in[0].normal[0]=1;in[0].matchingDistance=1.5;in[0].ownerDistance=.5;
in[0].quadrature.volume=9;in[0].quadrature.firstMoment=-3;
std::memset(out,0xA5,sizeof(out));
ck(ugkwp::evaluateGasBoundaryLayerSlot(s,0),"33 actual matching donors rejected");
ck(exchange[0].ready&&status[0]==0,"output is not completely ready");
ck(in[0].matching.temperature==expectedT&&in[0].pressure==expectedP,"donor loop omitted actual input");
ck(modelStatus[0].code==ugkwp::gaswall::WallCode::Success,"convergence poisoned completed output");
ck(ugkwp::gasBoundaryLayerSlotReady(s,0),"bulk readiness rejected complete output");
ck(closure[0].volume==9&&closure[0].ownerDistance==Real(.5),"SST publication absent");
ck(speciesFlux[0]==-out[0].wallSpeciesFlux[0]&&speciesFlux[1]==-out[0].wallSpeciesFlux[1],"species publication absent");
ck(std::isfinite(exchange[0].energy)&&exchange[0].energy!=0,"energy publication absent");
// Finite donor metadata does not veto the actual donor arithmetic.
weights[0]=Real(-.1);owners[1]=0;
ck(ugkwp::evaluateGasBoundaryLayerSlot(s,0),"signed non-normalized donor weights rejected");
// Actual inaccessible data remains a specific failure, never a default result.
m.matchingCells=nullptr;ck(!ugkwp::evaluateGasBoundaryLayerSlot(s,0),"missing donor storage accepted");
ck(status[0]==int(ugkwp::GasTransportCode::InvalidStorage)&&!exchange[0].ready,"wrong missing storage status");
m.matchingCells=matching;matching[0]=34;ck(!ugkwp::evaluateGasBoundaryLayerSlot(s,0),"unaddressable donor accepted");
ck(status[0]==int(ugkwp::GasTransportCode::InvalidStorage)&&!exchange[0].ready,"wrong donor address status");
delete[] workspace;
}
'''

@pytest.mark.parametrize('sanitizers',[False,True])
def test_actual_donors_publish_complete_stage(tmp_path,sanitizers):
    source=fixture().replace('struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};',
        '#include "gasTransport/GasBoundaryLayerModelState.H"\nstruct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;};')
    compile_probe(tmp_path,source+BODY,sanitizers=sanitizers)


def test_original_species_scaling_overflow_preserves_complete_exchange(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
int main(){State s;initialise(s);auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
int status[1]={0};ugkwp::GasBoundaryLayerExchange<Real> exchange[1];
ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real flux[2]={123,456};
w.status=status;w.exchange=exchange;w.sst=closure;w.speciesFlux=flux;
ugkwp::gaswall::WallInput<Real,2> in;in.pressure=Real(1e-300);in.normal[0]=1;in.ownerDistance=1;in.quadrature.volume=1;in.model.thermo=s.gasSpecies.thermo;
auto*coeff=const_cast<Real*>(in.model.thermo.coefficients);coeff[0]=coeff[3]=1000;coeff[2]=coeff[5]=0;
ugkwp::gaswall::WallOutput<Real,2> out;out.traceTemperature=300;out.wallSpeciesFlux[0]=2;out.wallSpeciesFlux[1]=-2;
ck(ugkwp::publishGasBoundaryLayerOutput(s,0,in,out,Real(1e308),Real(0)),"original complete exchange refused");
ck(exchange[0].ready&&std::isinf(flux[0])&&flux[0]<0&&std::isinf(flux[1])&&flux[1]>0,"original species arithmetic changed");
}
''')
