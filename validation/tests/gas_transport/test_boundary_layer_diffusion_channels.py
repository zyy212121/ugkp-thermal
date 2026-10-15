"""Real host-emulated kernels: ordinary masks must preserve wall-slot transport.

These checks retain the existing coefficient bound, not a proof of nonlinear
wall-profile stability or native GPU execution.
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


def test_ordinary_boundary_bound_uses_actual_diffusion_channels(tmp_path):
    compile_probe(tmp_path, fixture()+CHECK+r'''
int main(){
 const int cases[][3]={{1,0,0},{1,1,0},{2,0,0},{2,1,0},{0,0,0},{0,1,0},{0,0,1},{3,0,0},{4,0,0}};
 for(const auto& row:cases){State s;initialise(s);const int kind=row[0],fixed=row[1],composition=row[2];
  s.gasMu=Real(.01);s.gasThermalConductivity=Real(.5);s.maxDiffusionNumber=1;
  Real D[2]={Real(2),Real(3)};s.gasSpecies.diffusivity=D;
  s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.deltaCoeffs[1]=Real(1000);
  s.riemannBoundaryKind[1]=kind;s.riemannBoundaryTFix[1]=fixed;s.riemannBoundaryT[1]=600;
  s.gasSpecies.compositionBoundaryFixed[1]=composition;
  s.gasSpecies.boundaryMassFraction[1]=Real(.2);s.gasSpecies.boundaryMassFraction[4]=Real(.8);
  threadIdx.x=0;computeGasDiffusionNumberKernel(&s,1.,Real(1));
  const Real nu=kind==1||kind==3||kind==4?Real(0):s.gasMu/s.rho[0];
  const Real cv=s.gasSpecies.heatCapacity[0]-s.gasSpecies.gasConstant[0];
  const Real alpha=fixed&&kind!=3&&kind!=4?s.gasThermalConductivity/(s.rho[0]*cv):Real(0);
  const Real species=kind==0&&composition?Real(6):Real(0);
  const Real expected=std::max(nu,std::max(alpha,species))*s.magSf[1]*s.deltaCoeffs[1]/s.V[0];
  near(s.gasDiffusionNumber[0],expected,"ordinary boundary included an inactive diffusion channel");
  threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1.);
  ck(s.gasSpecies.faceStatus[1]==0,"physical boundary status");
  if(kind==1){ck(s.gasPhiRho[1]==0&&s.gasSpecies.flux[1]==0&&s.gasSpecies.flux[4]==0,"slip species impermeability");
   ck(fixed?s.gasPhiRhoE[1]!=0:s.gasPhiRhoE[1]==0,"slip thermal channel mismatch");}
 }
}
''')


@pytest.mark.parametrize('channel', ['thermal', 'species'])
def test_selected_wall_keeps_bound_without_ordinary_flags_or_ready_profile(tmp_path, channel):
    compile_probe(tmp_path, fixture()+CHECK+r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
int main(){for(int turbulent:{0,1}){
 State s;initialise(s);s.gasMu=Real(1e-8);s.gasThermalConductivity=THERMAL?Real(.5):Real(.001);
 s.maxDiffusionNumber=Real(.25);s.sstWallTreatment=2;s.turbulenceModel=turbulent?3:0;s.sstConfigured=turbulent;
 Real D[2]={THERMAL?Real(0):Real(.02),THERMAL?Real(0):Real(.03)};s.gasSpecies.diffusivity=D;
 s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.deltaCoeffs[1]=Real(1000);s.magSf[1]=2;s.Sfx[1]=-2;
 s.riemannBoundaryKind[1]=2;s.riemannBoundaryTFix[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=0;
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;
 int faces[3]={-1,0,-1},owners[2]={0,-1},status[1]={};w.faceSlot=faces;w.ownerSlot=owners;w.status=status;
 ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real speciesFlux[2]={};
 w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
 const Real cv=s.gasSpecies.heatCapacity[0]-s.gasSpecies.gasConstant[0];
 const Real nu=s.gasMu/s.rho[0],alpha=s.gasThermalConductivity/(s.rho[0]*cv);
 const Real species=Real(2)*std::max(D[0],D[1]);
 const Real weight=Real(.5)*Real(.1)/s.maxDiffusionNumber*s.magSf[1]*s.deltaCoeffs[1]/s.V[0];
 const Real expected=std::max(nu,std::max(alpha,species))*weight;
 ck(expected>nu*weight,"test must make the wall-only thermal/species channel dominant");
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
 threadIdx.x=0;computeGasDiffusionNumberKernel(&s,.1,Real(.5));
 near(s.gasDiffusionNumber[0],expected,"ready wall lost its thermal/species bound");
 const Real after[]={exchange[0].mass,exchange[0].momentumX,exchange[0].momentumY,exchange[0].momentumZ,
  exchange[0].energy,exchange[0].k,exchange[0].matchingPressure,speciesFlux[0],speciesFlux[1]};
 ck(std::memcmp(saved,after,sizeof saved)==0&&exchange[0].ready&&exchange[0].massReady,"bound mutated physical wall exchange");
 // Cold outer estimation must retain the bound without demanding a BVP solve.
 exchange[0].ready=false;computeGasDiffusionNumberKernel(&s,.1,Real(.5));
 near(s.gasDiffusionNumber[0],expected,"selected cold wall lost its thermal/species bound");
 ck(!exchange[0].ready&&status[0]==0&&s.gasSpecies.faceStatus[1]==0,"outer estimate forced or changed profile readiness");
 // Enablement and face selection, not a wall-treatment label, control the path.
 w.enabled=false;computeGasDiffusionNumberKernel(&s,.1,Real(.5));
 near(s.gasDiffusionNumber[0],nu*weight,"disabled wall incorrectly retained wall-only channels");
 w.enabled=true;faces[1]=-1;computeGasDiffusionNumberKernel(&s,.1,Real(.5));
 near(s.gasDiffusionNumber[0],nu*weight,"unselected wall incorrectly retained wall-only channels");
}}
'''.replace('THERMAL', str(int(channel == 'thermal'))))


@pytest.mark.parametrize('strict',[False,True])
def test_inactive_channels_still_require_consumed_volume_storage(tmp_path,strict):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);s.maxDiffusionNumber=1;
 s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.gasGeometry.enabled=true;s.gasGeometry.interval=1;
 threadIdx.x=0;computeGasDiffusionNumberKernel(&s,1.,Real(1),STRICT);
 ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage)&&std::isnan(s.gasDiffusionNumber[0]),"inactive channel omitted required volume backing error");
}
'''.replace('STRICT','true' if strict else 'false'))
