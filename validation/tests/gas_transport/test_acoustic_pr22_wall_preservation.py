"""Acoustic-only timestep changes retain PR22 conservative and macro wall physics."""
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_numerical_wall_k_inventory import SETUP


def test_conservative_wall_sources_and_sst_face_flux_remain_active(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
#include "gasTransport/GasConservativeWall.H"
struct ConservativeState:State {
 struct {ugkwp::ConservativeWallData conservative;} gasBoundaryLayerModel;
};
int main(){ConservativeState s;initialise(s);s.sstConfigured=1;
int owners[2]={0,-1},faces[3]={-1,0,-1},indices[3]={0,1,2},active[1]={1},status[1]={0};
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];exchange[0].ready=true;
ugkwp::GasBoundaryLayerSstClosure<Real> closures[1];
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=faces;w.ownerSlot=owners;w.status=status;w.exchange=exchange;w.sst=closures;
Real sourceK[2]={2.5,0},sourceOmega[2]={7.25,0};
ugkwp::ConservativeWallFace flux[3];
auto&d=s.gasBoundaryLayerModel.conservative;d.enabled=true;d.cellSlot=owners;d.faceSlot=faces;d.faceIndex=indices;d.active=active;d.face=flux;d.sourceK=sourceK;d.sourceOmega=sourceOmega;
Real k=0,o=0;ck(sstSourcesForCell(s,0,Real(999),k,o),"conservative source unavailable");
near(k,sourceK[0],"conservative k volume source replaced");near(o,sourceOmega[0],"conservative omega volume source replaced");
flux[1].orientation=-1;flux[1].kUpwind=.25;flux[1].omegaUpwind=.75;flux[1].diffusion[5]=7;flux[1].diffusion[6]=11;s.gasPhiRho[1]=5;
threadIdx.x=1;computeSstFaceFluxKernel(&s);
near(s.sstPhiRhoK[1],Real(5)*Real(.25)-7,"conservative k face flux replaced");
near(s.sstPhiRhoOmega[1],Real(5)*Real(.75)-11,"conservative omega face flux replaced");
}
''')


def test_macro_inventory_omits_profile_volume_source_not_physical_wall_flux(tmp_path):
    setup=SETUP.replace('in.ownerDistance=', 'in.volumeSstClosure=true;s.gasBoundaryLayer.useMacroSstInventory=true;in.ownerDistance=')
    compile_probe(tmp_path, fixture()+setup+r'''
int main(){State s;initialise(s);wall(s,Real(4),Real(100));
const auto& closure=s.gasBoundaryLayer.sst[0];
ck(closure.kFluxCount==0,"macro inventory published profile internal flux replacement");
ck(closure.nonDestructiveKSource==0&&closure.kDestructionRate==0,"macro inventory acquired profile volume source");
near(s.gasBoundaryLayer.exchange[0].k,Real(100)*s.magSf[1],"macro inventory lost BVP physical wall k exchange");
ck(s.gasBoundaryLayer.exchange[0].ready,"physical wall exchange not published");
}
''')
