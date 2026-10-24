"""Production gas/SST operators consume exactly one physical wall closure."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_sst_ale_audit import AUDIT

SETUP=r'''
void wall(State&s){s.sstWallTreatment=2;s.sstConfigured=1;s.turbulenceModel=3;s.riemannBoundaryKind[1]=2;
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=new int[3]{-1,0,-1};w.ownerSlot=new int[2]{0,-1};w.status=new int[1]{};
w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{};
w.exchange[0].matchingPressure=s.p[0];w.exchange[0].ready=true;w.exchange[0].mass=-.4;w.exchange[0].momentumX=-17;w.exchange[0].momentumY=3;w.exchange[0].energy=-31;w.exchange[0].k=-2;
s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;w.exchange[0].k=0;w.sst[0]={4,11,1,.5};}
'''

def test_selected_wall_uses_actual_k_once_and_independent_inventory_balance(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);wall(s);audit(s);
threadIdx.x=0;applySstWallFunctionStateKernel(&s);near(s.omega[0],11,"new owner omega not applied");near(s.rhoK[0],6,"k was projected");
threadIdx.x=1;computeSstFaceFluxKernel(&s);near(s.sstPhiRhoK[1],0,"modeled wall k flux not used");
s.sstPhiRhoK[0]=.5;s.sstPhiRhoOmega[0]=.7;
threadIdx.x=0;Real source,unused;sstSourcesForCell(s,0,Real(0),source,unused);applySstFluxAndSourceKernel(&s,.01);near(s.rhoK[0],Real(6)+Real(.01)*(source-Real(.5)),"wall source was not applied exactly once");recoverSstPrimitivesKernel(&s);
near(s.rhoOmega[0],22,"finite owner constraint not retained");near(observed.sourceK[0],Real(.01)*source,"test-side source impulse disagrees");
checkBudget(s,6,8,0);ck(observed.transportOmega[0]!=0,"replaced omega transport row omitted");ck(observed.sourceOmega[0]!=0,"replaced omega source row omitted");
Real saved=s.rhoOmega[0];recoverSstPrimitivesKernel(&s);near(s.rhoOmega[0],saved,"omega projection not idempotent");checkBudget(s,6,8,0);
}
''')

def test_physical_wall_flux_requires_ready_exchange(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);Real m,x,y,z,e;
ck(computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e),"ready physical wall rejected");near(m,Real(-.4),"wall mass missing");near(x,-17,"wall momentum missing");near(e,-31,"wall energy missing");
s.gasBoundaryLayer.exchange[0].ready=false;
ck(!computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e),"unprepared wall silently fell back");ck(s.gasSpecies.faceStatus[1]!=0,"unprepared wall did not reject trial");
}
''')

def test_owner_geometry_mismatch_rejects_before_inventory_write(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasBoundaryLayer.sst[0].volume=2;
threadIdx.x=0;Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);applySstFluxAndSourceKernel(&s,.01);ck(s.gasSpecies.cellStatus[0]==0,"finite volume certificate refused");near(s.rhoK[0],Real(6)+Real(.01)*sk,"actual wall source missing");near(s.rhoOmega[0],8,"wall omega constraint changed");}
''')

def test_large_finite_wall_source_updates_physical_inventory(tmp_path):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.k[0]=0.001;s.rhoK[0]=.002;s.omega[0]=1000;
threadIdx.x=0;Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);const Real expected=clampMin(s.rhoK[0]+Real(.1)*sk,s.rho[0]*s.sstKMin);applySstFluxAndSourceKernel(&s,.1);ck(s.gasSpecies.cellStatus[0]==0,"finite source diagnostic refused");near(s.rhoK[0],expected,"actual clamped k candidate missing");}
''')

def test_prescribed_wall_flux_is_rejected_rather_than_partially_scaled(tmp_path):
    from test_mixture_transport import fixture as mixture
    compile_probe(tmp_path,mixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasPhiRho[1]=0;s.gasPhiRhoE[1]=17;
s.gasSpecies.flux[1]=.5;s.gasSpecies.flux[4]=-.5;s.gasSpecies.positivityScale[0]=.2;
threadIdx.x=1;applyGasFluxPositivityScaleKernel(&s);
ck(s.gasSpecies.faceStatus[1]==0,"prescribed finite counterflow rejected");
ck(s.gasSpecies.flux[1]==.5&&s.gasSpecies.flux[4]==-.5&&s.gasPhiRhoE[1]==17,"fixed physical exchange was partially scaled");}
''')



def test_test_side_observer_does_not_change_physical_wall_update(tmp_path):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){Real accepted[2][2]{};for(int enabled=0;enabled<2;++enabled){State s;initialise(s);wall(s);audit(s);observed.enabled=enabled;
threadIdx.x=0;applySstWallFunctionStateKernel(&s);threadIdx.x=1;computeSstFaceFluxKernel(&s);
s.sstPhiRhoK[0]=.5;s.sstPhiRhoOmega[0]=.7;threadIdx.x=0;Real source,unused;sstSourcesForCell(s,0,Real(0),source,unused);applySstFluxAndSourceKernel(&s,.01);recoverSstPrimitivesKernel(&s);
near(s.rhoK[0],Real(6)+Real(.01)*(source-Real(.5)),"test observer changed physical wall k source");near(s.rhoOmega[0],22,"test observer changed owner omega");
accepted[enabled][0]=s.rhoK[0];accepted[enabled][1]=s.rhoOmega[0];if(enabled)checkBudget(s,6,8,0);
}ck(accepted[0][0]==accepted[1][0]&&accepted[0][1]==accepted[1][1],"test-side observation changed physical result");}
''')
