"""Invalid wall exchange is diagnosed before any invalid publication."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

def test_wall_publication_preserves_first_failure_and_old_flux(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
int main(){
 State s;initialise(s);ugkp::diagnostics::DeviceContext diagnostics{};
 s.diagnostics=&diagnostics;s.gasSpecies.thermo.diagnostics=&diagnostics;
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
 int status[1]={0};w.status=status;
 ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];
 Real speciesFlux[2]={17,19};w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
 ugkwp::gaswall::WallInput<Real,2> input;input.pressure=100;input.normal[0]=1;input.model.thermo=s.gasSpecies.thermo;
 input.quadrature.volume=3;input.ownerDistance=.5;
 ugkwp::gaswall::WallOutput<Real,2> output;output.traceTemperature=700;output.traceVelocity[0]=2;
 output.wallSpeciesFlux[0]=.4;output.wallSpeciesFlux[1]=-.1;output.conductiveHeatFlux=50;output.ownerOmega=11;
 ugkp::diagnostics::reset(diagnostics,9,4,7,.25);
 ck(ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(2),Real(.25)),"finite exchange changed");
 const Real oldFlux=speciesFlux[0],oldEnergy=exchange[0].energy;
 const auto nan=std::numeric_limits<Real>::quiet_NaN();
 ck(!ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,nan,Real(.25)),"NaN area published");
 ck(diagnostics.fatalClaim&&diagnostics.firstFatal.sourceSiteId==2501,"missing area fatal");
 ck(diagnostics.firstFatal.step==7&&diagnostics.firstFatal.simulationTime==.25,"source operation lost");
 ck(!exchange[0].ready&&!exchange[0].massReady&&exchange[0].energy==oldEnergy&&speciesFlux[0]==oldFlux,"invalid exchange changed storage");
 ugkp::diagnostics::reset(diagnostics,10,4,8,.5);output.traceTemperature=-1;
 ck(!ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(2),Real(.25)),"negative temperature published");
 ck(diagnostics.firstFatal.code==ugkp::diagnostics::Code::NegativeTemperature,"negative temperature misclassified");
 ugkp::diagnostics::reset(diagnostics,11,4,9,.75);output.traceTemperature=700;output.traction[1]=nan;
 ck(!ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,Real(2),Real(.25)),"NaN traction published");
 ck(diagnostics.firstFatal.sourceSiteId==10057&&diagnostics.firstFatal.component==4,"traction source lost");
}
''')
