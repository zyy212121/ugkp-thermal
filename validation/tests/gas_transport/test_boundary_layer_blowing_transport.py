
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_boundary_layer_operators import SETUP

def test_blowing_macro_interface_uses_bvp_endpoint_k(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r"""
#include "gasWall/WallModel.H"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
void near(Real a,Real b,const char* m){if(std::abs(a-b)>1e-9){std::fprintf(stderr,"%s: %.17g vs %.17g\n",m,double(a),double(b));std::abort();}}
int main(){State s;initialise(s);wall(s);
using namespace ugkwp;using namespace ugkwp::gaswall;
WallInput<Real,2> in;in.model.thermo=s.gasSpecies.thermo;
in.model.viscosity=2e-5;in.model.conductivity=.04;
in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
in.pressure=101325;in.temperature=in.matching.temperature=500;
in.normal[0]=1;in.ownerDistance=.0005;in.matchingDistance=.001;
in.matching.velocity[1]=30;in.matching.k=.5;in.matching.omega=2000;
in.matching.massFraction[0]=.9;in.matching.massFraction[1]=.1;
in.massFlux[0]=.00018;in.massFlux[1]=.00002;
Real y[2]={.00025,.00075},weight[2]={.0005,.0005};
in.quadrature={y,weight,2,.001,5e-7};
in.volumeSstClosure=true;in.integrateOwnerKSource=true;
WallModelConfig<Real> cfg;cfg.nodes=24;cfg.enableSst=true;
WallWorkspace<Real,2,24> workspace;WallOutput<Real,2> output;WallStatus status;
ck(evaluateWallModel(in,cfg,workspace,output,status),"BVP solve failed");
ck(publishGasBoundaryLayerOutput(s,0,in,output,Real(1),Real(0)),"publication failed");
s.Cx[0]=.0005;s.Cx[1]=.0015;s.faceCx[0]=.001;s.k[0]=.1;s.k[1]=.5;
s.gasPhiRho[0]=.0002;
threadIdx.x=0;computeSstFaceFluxKernel(&s);
const Real expected=output.kTransportFlux[output.kFluxCount-1];
near(s.sstPhiRhoK[0],expected,"macro upwind k disagrees with BVP endpoint flux");
// Shared-face antisymmetry preserves global inventory; extra tangential/ALE
// mass transport must retain its ordinary upwind value.
s.gasPhiRho[0]=.0007;computeSstFaceFluxKernel(&s);
near(s.sstPhiRhoK[0],expected+Real(.0005)*s.k[0],"unmodeled mass transport overwritten");
}
""")
