"""The FV publisher keeps representative omega separate from k decay."""
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_numerical_wall_k_inventory import SETUP


def test_published_point_omega_does_not_change_projected_k_destruction(tmp_path):
    setup=SETUP.replace('in.ownerDistance=', 'in.volumeSstClosure=true;in.ownerDistance=')
    setup=setup.replace('out.ownerOmega=omega;', 'out.ownerKContent=3;out.ownerKDestruction=21;out.kFluxCount=3;out.ownerOmega=omega;')
    compile_probe(tmp_path,fixture()+setup+r"""
int main(){for(Real omega:{Real(10),Real(1000)}){
 State s;initialise(s);wall(s,Real(4),omega);auto& c=s.gasBoundaryLayer.sst[0];
 near(c.ownerOmega,omega,"publisher changed point omega");
 near(c.kDestructionRate,Real(7),"publisher conflated point value and destruction moment");
 threadIdx.x=0;applySstWallFunctionStateKernel(&s);
 near(s.omega[0],omega,"macro constraint lost representative omega");
 Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source failed");
 near(sk,c.nonDestructiveKSource-Real(7)*s.rhoK[0],"point omega changed volume k budget");
 near(sstKSourceActivity(s,0,sk,Real(0)),std::abs(c.nonDestructiveKSource)+Real(7)*s.rhoK[0],"source bound uses wrong decay");
}}
""")


def test_zero_profile_k_preserves_existing_macro_decay(tmp_path):
    setup=SETUP.replace('in.ownerDistance=', 'in.volumeSstClosure=true;in.ownerDistance=')
    setup=setup.replace('out.ownerOmega=omega;', 'out.ownerKContent=0;out.ownerKDestruction=0;out.kFluxCount=3;out.ownerOmega=omega;')
    compile_probe(tmp_path,fixture()+setup+r"""
int main(){State s;initialise(s);wall(s,Real(4),Real(100));
 near(s.gasBoundaryLayer.sst[0].kDestructionRate,Real(9),"zero profile lost existing macro decay");
 auto& c=s.gasBoundaryLayer.sst[0];Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source failed");
 near(sk,c.nonDestructiveKSource-Real(9)*s.rhoK[0],"zero profile froze nonzero macro k");}
""")
