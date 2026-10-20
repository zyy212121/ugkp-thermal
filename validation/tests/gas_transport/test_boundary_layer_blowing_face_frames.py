"""Normal transport replacement keeps a single conservative shared face."""
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_boundary_layer_operators import SETUP


def test_normal_transport_signs_and_ale_remainder_for_either_cell(tmp_path):
    compile_probe(tmp_path, fixture() + SETUP + r'''
int main(){
 // n, prescribed wall mass, model total k flux, actual ALE mass, expected k flux.
 const Real cases[][5]={{1,.4,1.2,.4,1.2},{1,-.4,-1.2,-.4,-1.2},
  {-1,.4,1.2,-.4,-1.2},{-1,-.4,-1.2,.4,1.2},
  {1,.4,1.2,-.6,-3.8},{-1,.4,1.2,.6,.8}};
 for(int side=0;side<2;++side)for(const auto& row:cases){State s;initialise(s);wall(s);
  s.gasMu=.25;s.k[0]=2;s.k[1]=5;s.rhoK[0]=4;s.rhoK[1]=10;
  int owners[2]={side==0?0:-1,side==1?0:-1};s.gasBoundaryLayer.ownerSlot=owners;
  Real heights[2]={0,2},kflux[2]={row[2],row[2]},oflux[2]={0,0};
  auto& c=s.gasBoundaryLayer.sst[0];c.normal[0]=row[0];c.normal[1]=0;
  c.kFluxCount=2;c.kFluxHeight=heights;c.kTransportFlux=kflux;c.omegaDiffusiveFlux=oflux;
  c.normalMassFlux=row[1];c.ownerDistance=1;s.gasPhiRho[0]=row[3];
  Real oldV[2]={1,1},newV[2]={1.1,.9},sweep[3]={.1,0,0};moving(s,oldV,newV,sweep,Real(.01));
  threadIdx.x=0;computeSstFaceFluxKernel(&s);
  near(s.sstPhiRhoK[0],row[4],"normal transport orientation or ALE remainder changed");
 }
}
''')


def test_two_wall_contributors_share_one_flux_and_global_moving_inventory(tmp_path):
    compile_probe(tmp_path, fixture() + SETUP + r'''
int main(){State s;initialise(s);wall(s);s.gasMu=.25;
 auto& w=s.gasBoundaryLayer;w.count=2;
 int faceSlots[3]={-1,0,1},owners[2]={0,1},statuses[2]={0,0};
 ugkwp::GasBoundaryLayerExchange<Real> exchange[2];
 ugkwp::GasBoundaryLayerSstClosure<Real> closures[2];
 w.faceSlot=faceSlots;w.ownerSlot=owners;w.status=statuses;w.exchange=exchange;w.sst=closures;
 s.riemannBoundaryKind[1]=s.riemannBoundaryKind[2]=2;
 Real heights[2]={0,2},positive[2]={1.2,1.2},negative[2]={-1.2,-1.2},omega[2]={0,0};
 for(int side=0;side<2;++side){
  exchange[side].ready=true;exchange[side].k=side==0?-.1:.2;
  auto& c=closures[side];c.normal[0]=side==0?1:-1;c.normalMassFlux=side==0?.4:-.4;
  c.ownerDistance=1;c.volume=1;c.ownerOmega=4;c.nonDestructiveKSource=0;
  c.kFluxCount=2;c.kFluxHeight=heights;c.kTransportFlux=side==0?positive:negative;c.omegaDiffusiveFlux=omega;
 }
 s.k[0]=2;s.k[1]=5;s.rhoK[0]=4;s.rhoK[1]=10;s.gasPhiRho[0]=.4;
 Real oldV[2]={1,1},newV[2]={1.1,.9},sweep[3]={.1,0,0};const Real dt=.001;moving(s,oldV,newV,sweep,dt);
 for(int face=0;face<3;++face){threadIdx.x=face;computeSstFaceFluxKernel(&s);}
 near(s.sstPhiRhoK[0],Real(1.2),"two agreeing profiles counted twice");
 near(s.sstPhiRhoK[1],Real(-.1),"left physical wall k sign");
 near(s.sstPhiRhoK[2],Real(.2),"right physical wall k sign");
 Real sources=0;for(int cell=0;cell<2;++cell){Real sk,so;ck(sstSourcesForCell(s,cell,Real(0),sk,so),"source rejected");sources+=sk;}
 for(int cell=0;cell<2;++cell){threadIdx.x=cell;applySstFluxAndSourceKernel(&s,dt);}
 near(s.rhoK[0]*newV[0]+s.rhoK[1]*newV[1],Real(14)+dt*(sources-Real(.1)),"shared face created global k inventory");
}
''')
