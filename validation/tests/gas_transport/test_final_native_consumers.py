"""Current wall publication -> real final face operator, host CUDA arithmetic.

Historical retry/terminal-NaN acceptance is not the resident backend contract.
The shared producer must reject invalid candidates; a final face cannot consume
an unpublished exchange. Pressure remains the local reconstructed pressure.
"""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

SETUP = r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
struct WallProbe {
 int slots[3]={-1,0,-1},owners[2]={0,-1},status[1]={0};
 ugkwp::GasBoundaryLayerExchange<Real> exchange[1];
 ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real species[2]{};
 ugkwp::gaswall::WallInput<Real,2> input;ugkwp::gaswall::WallOutput<Real,2> output;
 void attach(State&s){auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
  w.faceSlot=slots;w.ownerSlot=owners;w.status=status;w.exchange=exchange;w.sst=closure;w.speciesFlux=species;
  s.gasBoundaryKind[1]=s.riemannBoundaryKind[1]=2;
  input.pressure=s.p[0]+17;input.normal[0]=1;input.model.thermo=s.gasSpecies.thermo;
  input.ownerDistance=.5;input.quadrature.volume=1;
  output.traceTemperature=700;output.traceVelocity[0]=.2;output.traceVelocity[1]=3;
  output.wallSpeciesFlux[0]=.004;output.wallSpeciesFlux[1]=.001;
  output.conductiveHeatFlux=50;output.traction[1]=5;output.ownerOmega=11;
 }
};
'''


def test_final_face_consumes_published_species_heat_and_local_pressure(tmp_path):
    compile_probe(tmp_path, fixture()+SETUP+r'''
int main(){State s;initialise(s);WallProbe wall;wall.attach(s);
 ck(ugkwp::publishGasBoundaryLayerOutput(s,0,wall.input,wall.output,Real(1),Real(0)),"publication failed");
 Real f[5]{};ck(computeRiemannGasFaceFluxDevice<false>(s,1,f[0],f[1],f[2],f[3],f[4],true),"final flux rejected");
 auto&e=wall.exchange[0];
 ck(f[0]==e.mass&&f[2]==e.momentumY&&f[3]==e.momentumZ&&f[4]==e.energy,"final face changed physical exchange");
 ck(std::abs(f[1]-(e.momentumX+17))<1e-9,"local pressure correction missing");
 for(int j=0;j<2;++j)ck(s.gasSpecies.flux[j*s.nFaces+1]==wall.species[j],"final species not published");
 ck(std::abs(f[0]-wall.species[0]-wall.species[1])<1e-15,"mass/species mismatch");
}
''')


@pytest.mark.parametrize("invalid", ["temperature", "heat", "species", "traction"])
def test_rejected_wall_candidate_cannot_feed_final_operator(tmp_path,invalid):
    field={"temperature":"traceTemperature", "heat":"conductiveHeatFlux",
           "species":"wallSpeciesFlux[0]", "traction":"traction[1]"}[invalid]
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);WallProbe wall;wall.attach(s);
 ck(ugkwp::publishGasBoundaryLayerOutput(s,0,wall.input,wall.output,Real(1),Real(0)),"initial publication failed");
 const Real accepted=wall.exchange[0].energy;const Real energy=s.rhoE[0],inventory=s.gasSpecies.rho[0];
 wall.output.FIELD=std::numeric_limits<Real>::quiet_NaN();
 ck(!ugkwp::publishGasBoundaryLayerOutput(s,0,wall.input,wall.output,Real(1),Real(0)),"invalid candidate accepted");
 ck(!wall.exchange[0].ready&&wall.exchange[0].energy==accepted,"failed candidate overwritten accepted value");
 Real f[5]{};ck(!computeRiemannGasFaceFluxDevice<false>(s,1,f[0],f[1],f[2],f[3],f[4],true),"unpublished exchange consumed");
 ck(s.gasSpecies.faceStatus[1]!=0,"missing failure diagnosis");
 ck(s.rhoE[0]==energy&&s.gasSpecies.rho[0]==inventory,"failed face changed inventory");
}
'''.replace("FIELD",field))
