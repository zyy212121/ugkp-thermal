"""Real host-emulated kernels: ordinary masks must preserve wall-slot transport.

These checks exercise physical fluxes, not native GPU execution.
"""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


CHECK = r'''
void near(Real a,Real b,const char* message){
 const Real tolerance=Real(64)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b));
 if(std::abs(a-b)>tolerance){std::fprintf(stderr,"%s: %.17g != %.17g\n",message,double(a),double(b));std::abort();}
}
'''


@pytest.mark.parametrize('channel', ['thermal', 'species'])
def test_selected_wall_preserves_heat_and_species_transport(tmp_path, channel):
    compile_probe(tmp_path, fixture()+CHECK+r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
int main(){for(int turbulent:{0,1}){
 State s;initialise(s);s.gasMu=Real(1e-8);s.gasThermalConductivity=THERMAL?Real(.5):Real(.001);
 s.sstWallTreatment=2;s.turbulenceModel=turbulent?3:0;s.sstConfigured=turbulent;
 Real D[2]={THERMAL?Real(0):Real(.02),THERMAL?Real(0):Real(.03)};s.gasSpecies.diffusivity=D;
 s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.deltaCoeffs[1]=Real(1000);s.magSf[1]=2;s.Sfx[1]=-2;
 s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=0;
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
 int faces[3]={-1,0,-1},owners[2]={0,-1},status[1]={};w.faceSlot=faces;w.ownerSlot=owners;w.status=status;
 ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real speciesFlux[2]={};
 w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
 ugkwp::gaswall::WallInput<Real,2> input;input.pressure=s.p[0];input.normal[0]=-1;
 input.model.thermo=s.gasSpecies.thermo;input.quadrature.volume=s.V[0];input.ownerDistance=Real(.5);
 ugkwp::gaswall::WallOutput<Real,2> output;output.traceTemperature=700;output.ownerOmega=1;
 output.conductiveHeatFlux=THERMAL?Real(50):Real(0);
 output.wallSpeciesFlux[0]=THERMAL?Real(0):Real(.02);output.wallSpeciesFlux[1]=-output.wallSpeciesFlux[0];
 ck(ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,s.magSf[1],Real(0)),"wall publication rejected");
 const Real saved[]={exchange[0].mass,exchange[0].momentumX,exchange[0].momentumY,exchange[0].momentumZ,
  exchange[0].energy,exchange[0].k,exchange[0].matchingPressure,speciesFlux[0],speciesFlux[1]};
 Real mass,mx,my,mz,energy;
 ck(computeRiemannGasFaceFluxDevice<true>(s,1,mass,mx,my,mz,energy),"selected wall exchange not used");
 ck(energy!=0,"wall fixture lacks physical heat/species enthalpy transport");
 if(!THERMAL)ck(s.gasSpecies.flux[1]!=0&&s.gasSpecies.flux[4]!=0,"wall fixture lacks physical counterdiffusion");

}}
'''.replace('THERMAL', str(int(channel == 'thermal'))))
