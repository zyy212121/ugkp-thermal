
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_boundary_layer_operators import SETUP

def test_multidimensional_production_includes_cross_strain(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+'\n#include "gasWall/WallModel.H"\n'+r"""
int main(){State s;initialise(s);wall(s);
s.rho[0]=1;s.k[0]=100;s.rhoK[0]=100;s.omega[0]=1;s.nut[0]=1;
auto& c=s.gasBoundaryLayer.sst[0];c.kFluxCount=3;c.normal[1]=1;
c.ownerOmega=0;
s.gradUxY[0]=1;
for(Real transverse:{Real(-1),Real(1)}){
 s.gradUyX[0]=transverse;
 ugkwp::gaswall::WallInput<Real,2> in;
 in.model.viscosity=.01;in.model.sstCoefficients=s.sstCoefficients;
 in.model.sstCoefficients.a1=100; // exact nut=k/omega=1, no viscosity clipping
 in.model.sstCoefficients.c1=1000;
 in.sstTangentialGradient[2][0]=transverse;
 ugkwp::gaswall::detail::LayerPoint<Real,2> p;p.rho=1;p.k=1;p.omega=1;p.z=1;
 ugkwp::gaswall::detail::applySstGradients(in,Real(1),Real(2),Real(0),Real(0),Real(0),Real(0),p);
 c.nonDestructiveKSource=p.sourceK+in.model.sstCoefficients.betaStar;c.kDestructionRate=0; // isolate production, independently of point omega
 near(p.nut,Real(1),"fixture viscosity differs");
 Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source rejected");
 std::fprintf(stderr,"cross=%g production=%g expected=%g\n",double(transverse),double(sk),transverse<0?1.:9.);
 near(sk,transverse<0?Real(1):Real(9),"cross strain was omitted");
}}
""")


def test_complete_strain_is_limited_once_and_rotationally_consistent(tmp_path):
    compile_probe(tmp_path,r"""
#include "gasWall/WallModel.H"
int main(){
 using namespace ugkwp::gaswall;
 WallInput<Real,2> in;in.model.viscosity=.01;
 in.model.sstCoefficients.a1=100;
 in.model.sstCoefficients.c1=50; // cap 4.5, between cancelled/reinforced P
 for(Real cross:{Real(-1),Real(1)}){
  in.sstTangentialGradient[2][0]=cross;
  detail::LayerPoint<Real,2> p;p.rho=1;p.k=1;p.omega=1;p.z=1;
  detail::applySstGradients(in,Real(1),Real(2),Real(0),Real(0),Real(0),Real(0),p);
  const Real want=cross<0?Real(.91):Real(4.41);
  if(std::abs(p.sourceK-want)>1e-12)return 1;
 }
 // Pure rigid rotation must produce no turbulent kinetic energy.
 in.sstTangentialGradient[2][0]=-2;
 detail::LayerPoint<Real,2> p;p.rho=1;p.k=1;p.omega=1;p.z=1;
 detail::applySstGradients(in,Real(1),Real(2),Real(0),Real(0),Real(0),Real(0),p);
 if(std::abs(p.sourceK+Real(.09))>1e-12)return 2;
 // Isotropic expansion has zero deviatoric strain and only -2/3*k*div(U).
 in.sstTangentialGradient[2][0]=0;
 in.sstTangentialGradient[0][0]=in.sstTangentialGradient[1][1]=1;
 p={};p.rho=1;p.k=1;p.omega=1;p.z=1;
 detail::applySstGradients(in,Real(1),Real(0),Real(0),Real(1),Real(0),Real(0),p);
 if(std::abs(p.sourceK+Real(2.09))>1e-12)return 3;
}
""")
