"""Production host kernels: inactive channels must not constrain a timestep."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


@pytest.mark.parametrize('kind,fixed,composition',[(1,0,0),(1,1,0),(2,0,0),(2,1,0),(0,0,0),(0,1,0),(0,0,1),(3,0,0),(4,0,0)])
def test_boundary_bound_matches_physical_diffusion_channels(tmp_path,kind,fixed,composition):
    body=fixture()+r'''
int main(){State s;initialise(s);
const int kind=KIND,fixed=FIXED,composition=COMPOSITION;
s.gasMu=Real(.01);s.gasThermalConductivity=Real(.5);s.maxDiffusionNumber=1;
Real D[2]={Real(2),Real(3)};s.gasSpecies.diffusivity=D;
s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.deltaCoeffs[1]=Real(1000);
s.riemannBoundaryKind[1]=kind;s.riemannBoundaryTFix[1]=fixed;s.riemannBoundaryT[1]=600;
s.gasSpecies.compositionBoundaryFixed[1]=composition;
s.gasSpecies.boundaryMassFraction[1]=Real(.2);s.gasSpecies.boundaryMassFraction[4]=Real(.8);
threadIdx.x=0;computeGasDiffusionNumberKernel(&s,1.,Real(1));
const Real nu=kind==1||kind==3||kind==4?Real(0):s.gasMu/s.rho[0];
const Real alpha=fixed && kind!=3 && kind!=4?s.gasThermalConductivity/(s.rho[0]*(s.gasSpecies.heatCapacity[0]-s.gasSpecies.gasConstant[0])):Real(0);
const Real species=kind==0&&composition?Real(6):Real(0);
const Real expected=std::max(nu,std::max(alpha,species))*s.magSf[1]*s.deltaCoeffs[1]/s.V[0];
ck(std::abs(s.gasDiffusionNumber[0]-expected)<=Real(64)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),expected),"inactive boundary transport channel entered CFL");
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1.);
ck(s.gasSpecies.faceStatus[1]==0,"physical boundary status");
if(kind==1){ck(s.gasPhiRho[1]==0&&s.gasSpecies.flux[1]==0&&s.gasSpecies.flux[4]==0,"slip species impermeability");
 ck(fixed?s.gasPhiRhoE[1]!=0:s.gasPhiRhoE[1]==0,"slip thermal channel mismatch");}
}
'''.replace('KIND',str(kind)).replace('FIXED',str(fixed)).replace('COMPOSITION',str(composition))
    compile_probe(tmp_path,body)


def test_internal_species_and_invalid_ale_geometry_keep_limits(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.maxDiffusionNumber=1;Real D[2]={Real(.02),Real(.03)};s.gasSpecies.diffusivity=D;
s.cellPlaneCount[0]=1;s.cellFaceId[0]=0;threadIdx.x=0;computeGasDiffusionNumberKernel(&s,1.,Real(1));
ck(std::abs(s.gasDiffusionNumber[0]-Real(.06))<Real(1e-7),"internal species bound removed");
s.gasGeometry.enabled=true;s.gasGeometry.interval=1; // invalid geometry must fail before a zero-channel shortcut
s.cellFaceId[0]=1;computeGasDiffusionNumberKernel(&s,1.,Real(1));
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidGeometry)&&s.gasDiffusionNumber[0]==OfGreat,"inactive channel bypassed ALE geometry gate");
}
''')
