from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_boundary_layer_operators import SETUP

def test_bvp_uses_volume_source_and_shared_normal_diffusion(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r"""
int main(){State s;initialise(s);wall(s);s.gasMu=.01;
Real height[3]={0,1,2},kflux[3]={-2,-3,-4},oflux[3]={-5,-7,-9};
auto& closure=s.gasBoundaryLayer.sst[0];closure.normal[0]=1;
closure.kFluxCount=3;closure.kFluxHeight=height;closure.kTransportFlux=kflux;closure.omegaDiffusiveFlux=oflux;
closure.nonDestructiveKSource=10;
Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"volume source unavailable");
near(sk,Real(10)-s.sstCoefficients.betaStar*s.rhoK[0]*closure.ownerOmega,"second wall-function production remained");
s.gasPhiRho[0]=.4;threadIdx.x=0;computeSstFaceFluxKernel(&s);
near(s.sstPhiRhoK[0],Real(.4)*s.k[0]-Real(3),"BVP normal k flux absent");
near(s.sstPhiRhoOmega[0],Real(.4)*s.omega[0]-Real(7),"BVP omega gradient was replaced by effective-omega secant");
// A shared face is consumed with opposite orientation; no cell-local hidden
// diffusion source may create or destroy total k inventory.
s.gasBoundaryLayer.exchange[0].k=0;s.gasPhiRho[0]=0;
for(int f=0;f<3;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
Real source[2];for(int c=0;c<2;++c){threadIdx.x=c;ck(sstSourcesForCell(s,c,Real(0),source[c],so),"source failed");}
const Real old=s.rhoK[0]+s.rhoK[1],dt=1e-4;
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,dt);}
near(s.rhoK[0]+s.rhoK[1],old+dt*(source[0]+source[1]),"shared diffusion broke the global k budget");
}
""")


def test_volume_source_activity_bounds_heated_blowing_compression(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r"""
int main(){State s;initialise(s);wall(s);
auto& closure=s.gasBoundaryLayer.sst[0];closure.normal[0]=1;closure.kFluxCount=3;
for(Real projected:{Real(-100),Real(100)})for(Real div:{Real(-50),Real(0),Real(50)}){
 closure.nonDestructiveKSource=projected;Real sk,so;
 ck(sstSourcesForCell(s,0,div,sk,so),"volume source unavailable");
 Real activity=sstKSourceActivity(s,0,sk,div);
 ck(activity>=std::abs(sk)&&activity>=0,"projected compression hid actual source stiffness");
 Real decay=s.sstCoefficients.betaStar*s.rhoK[0]*closure.ownerOmega;
 ck(activity>=decay,"linear destruction vanished by cancellation");
}}
""")


def test_oblique_macro_face_quadrature_refines_at_second_order(tmp_path):
    # Wedge 0<x,z<1, 0<y<1+x and F(y)=max(0,y-1.5).
    # The exact top integral is 1/8. Odd midpoint partitions converge as h^2;
    # this explicitly exposes the remaining macro face-quadrature error.
    compile_probe(tmp_path,fixture()+SETUP+r"""
int main(){State s;initialise(s);wall(s);s.gasMu=.01;
Real height[3]={0,1.5,2},flux[3]={0,0,.5};
auto& c=s.gasBoundaryLayer.sst[0];c.normal[0]=0;c.normal[1]=1;
c.ownerDistance=Real(7)/9;c.kFluxCount=3;c.kFluxHeight=height;c.kTransportFlux=flux;c.omegaDiffusiveFlux=flux;
for(int n:{1,3,5,9}){Real integral=0,dx=Real(1)/n;
 for(int j=0;j<n;++j){s.faceCx[0]=(j+Real(.5))*dx;s.faceCy[0]=1+s.faceCx[0];
  s.Sfx[0]=-dx;s.Sfy[0]=dx;s.magSf[0]=std::sqrt(Real(2))*dx;
  s.Cx[0]=Real(5)/9;s.Cy[0]=Real(7)/9;s.Cx[1]=s.faceCx[0];s.Cy[1]=s.faceCy[0]+1;
  threadIdx.x=0;computeSstFaceFluxKernel(&s);integral+=s.sstPhiRhoK[0];
 }
 near(Real(.125)-integral,Real(.125)/(n*n),"oblique face quadrature lost second order");

}
}
""")
