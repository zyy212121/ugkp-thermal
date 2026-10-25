"""Pure diagnostic overflow must not reject finite SST/wall physical output."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_mixture_wall_function import wall_fixture


@pytest.mark.parametrize('mode', ['false', 'true'])
def test_finite_sst_inventory_needs_no_activity_ratio(tmp_path, mode):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
s.sstConfigured=1;s.sstWallTreatment=3;s.cellPlaneCount[0]=0;s.sstCoefficients=ugkwp::defaultSstCoefficients();
s.k[0]=s.rhoK[0]=1;s.omega[0]=s.rhoOmega[0]=std::numeric_limits<Real>::denorm_min()*Real(100);
s.sstOmegaMin=s.omega[0];s.sstKMin=1e-10;s.gradUxY[0]=1;s.nut[0]=.1;
s.gradKX[0]=1e-10;s.gradOmegaX[0]=1e-10;
Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source unavailable");
const Real expected=s.rhoOmega[0]+so;ck(std::isfinite(expected)&&expected>0,"physical result must remain finite");
threadIdx.x=0;applySstFluxAndSourceKernel(&s,1.,MODE);
ck(!ugkp::diagnostics::hasFatal(&d),"diagnostic activity overflow became fatal");
ck(s.rhoOmega[0]==expected&&s.gasSpecies.cellStatus[0]==0,"finite inventory was rejected");
}
'''.replace('MODE',mode))


@pytest.mark.parametrize('mode', ['false', 'true'])
def test_wall_raw_fraction_sum_overflow_matches_clipped_control(tmp_path, mode):
    compile_probe(tmp_path, wall_fixture()+r'''
int main(){State s,control;wall(s);wall(control);ugkp::diagnostics::DeviceContext d{},dc{};s.diagnostics=&d;control.diagnostics=&dc;
s.gasSpecies.rho[0]=s.gasSpecies.rho[2]=std::numeric_limits<Real>::max();
control.gasSpecies.rho[0]=control.gasSpecies.rho[2]=1;
Real mu=77,k=78,q=79,cmu=77,ckappa=78,cq=79;int active=1,ca=1;bool evaluated=false,ce=false;
gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,MODE,&evaluated);
gasFaceSubgridTransportProperties(control,1,0,-1,2,Real(1),cmu,ckappa,cq,ca,MODE,&ce);
ck(!ugkp::diagnostics::hasFatal(&d)&&evaluated,"raw audit sum overflow blocked wall transport");
ck(ce&&!ugkp::diagnostics::hasFatal(&dc),"clipped control unavailable");
ck(mu==cmu&&k==ckappa&&q==cq&&active==ca,"audit changed physical wall output");
ck(d.warnings[0].count>0&&!std::isfinite(d.warnings[0].sample.observed),"overflow conservation warning missing");
ck(d.warnings[2].count==2&&s.gasSpecies.faceStatus[1]==0,"fraction clipping or status changed");
}
'''.replace('MODE',mode))
