"""The actual stability kernel bounds the same source used by SST evolution."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_boundary_layer_operators import SETUP


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('closure', ['ready-volume', 'cold-volume', 'ready-nonvolume', 'coarse'])
def test_source_bound_tracks_production_source_without_duplicate_compression(tmp_path, direction, closure):
    setup = {
        'ready-volume': '',
        'cold-volume': 's.gasBoundaryLayer.exchange[0].ready=false;',
        'ready-nonvolume': 'closure.kFluxCount=0;',
        'coarse': 's.gasBoundaryLayer.enabled=false;s.sstWallTreatment=0;',
    }[closure]
    compile_probe(tmp_path, fixture() + SETUP + r'''
int main(){
State s;initialise(s);wall(s);threadIdx.x=0;
auto& closure=s.gasBoundaryLayer.sst[0];
closure.kFluxCount=3;closure.ownerOmega=s.omega[0];
closure.nonDestructiveKSource=s.rho[0]*s.sstCoefficients.betaStar*s.k[0]*s.omega[0];
SETUP
const Real dt=.1,targetCo=.5,divU=DIRECTION*Real(1000);
s.gasPhiRho[0]=divU*s.rho[0];s.gasPhiRho[1]=0;
// Obtain each endpoint from the production source itself. The stability
// kernel must bound these sources, rather than invent a second compression.
Real sourceAtZero,sourceAtDiv,unused;
ck(sstSourcesForCell(s,0,Real(0),sourceAtZero,unused),"zero-divergence source unavailable");
ck(sstSourcesForCell(s,0,divU,sourceAtDiv,unused),"finite-divergence source unavailable");
const Real activityAtZero=sstKSourceActivity(s,0,sourceAtZero,Real(0));
const Real activityAtDiv=sstKSourceActivity(s,0,sourceAtDiv,divU);
const Real expected=targetCo/s.sstMaxSourceNumber*dt
    *std::max(activityAtZero,activityAtDiv)/s.rhoK[0];
if(PROJECTED){
    near(sourceAtZero,Real(0),"balanced projected source at zero divergence");
    near(sourceAtDiv,Real(0),"projected source incorrectly depends on macro divergence");
    near(expected,Real(.144),"projected gross production/destruction activity");
}else{
    ck(std::abs(sourceAtDiv-sourceAtZero)>Real(1000),"coarse source lost its compression term");
    ck(expected>Real(100),"coarse source bound lost its compression activity");
}
computeSstStabilityNumberKernel(&s,dt,targetCo,REQUIRE);
ck(s.gasSpecies.cellStatus[0]==0,"source estimate diagnosed valid input");
near(s.sstSourceNumber[0],expected,"source stability differs from actual source activity");
}
'''.replace('SETUP', setup).replace('DIRECTION', str(direction))
        .replace('PROJECTED', 'true' if closure == 'ready-volume' else 'false')
        .replace('REQUIRE', 'false' if closure == 'cold-volume' else 'true'))