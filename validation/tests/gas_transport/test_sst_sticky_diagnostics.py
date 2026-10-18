"""SST physical failures must not publish invalid inventory in single pass."""
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture


def test_nonfinite_moving_sst_inventory_records_and_stops(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;s.sstConfigured=1;
 Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={0,0,0};moving(s,oldV,newV,sweep,Real(.001));
 s.sstPhiRhoK[0]=std::numeric_limits<Real>::quiet_NaN();const auto before=s.rhoK[0];
 threadIdx.x=0;applySstFluxAndSourceKernel(&s,Real(.001));
 ck(ugkp::diagnostics::hasFatal(&d),"SST physical failure not recorded");
 ck(s.rhoK[0]==before,"invalid moving SST inventory published");
}
''')
