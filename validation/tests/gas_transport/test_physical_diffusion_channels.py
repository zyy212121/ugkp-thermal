"""Production host kernels preserve physical diffusion channels."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


@pytest.mark.parametrize('kind,fixed,composition',[(1,0,0),(1,1,0),(2,0,0),(2,1,0),(0,0,0),(0,1,0),(0,0,1),(3,0,0),(4,0,0)])
def test_boundary_physical_diffusion_channels(tmp_path,kind,fixed,composition):
    body=fixture()+r'''
int main(){State s;initialise(s);
const int kind=KIND,fixed=FIXED,composition=COMPOSITION;
s.gasMu=Real(.01);s.gasThermalConductivity=Real(.5);
Real D[2]={Real(2),Real(3)};s.gasSpecies.diffusivity=D;
s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.deltaCoeffs[1]=Real(1000);
s.riemannBoundaryKind[1]=kind;s.riemannBoundaryTFix[1]=fixed;s.riemannBoundaryT[1]=600;
s.gasSpecies.compositionBoundaryFixed[1]=composition;
s.gasSpecies.boundaryMassFraction[1]=Real(.2);s.gasSpecies.boundaryMassFraction[4]=Real(.8);
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1.);
ck(s.gasSpecies.faceStatus[1]==0,"physical boundary status");
if(kind==1){ck(s.gasPhiRho[1]==0&&s.gasSpecies.flux[1]==0&&s.gasSpecies.flux[4]==0,"slip species impermeability");
 ck(fixed?s.gasPhiRhoE[1]!=0:s.gasPhiRhoE[1]==0,"slip thermal channel mismatch");}
}
'''.replace('KIND',str(kind)).replace('FIXED',str(fixed)).replace('COMPOSITION',str(composition))
    compile_probe(tmp_path,body)
