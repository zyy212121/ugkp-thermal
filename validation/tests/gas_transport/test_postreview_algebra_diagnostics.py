"""Physical faults are recorded before turbulence/reconstruction/gravity publication."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_authoritative_energy_sources import source_fixture

@pytest.mark.parametrize('fault,site', [
 ('s.smagorinskyCs=bad;',30326),('s.gradUxY[0]=bad;',30300),
 ('s.cellLength[0]=bad;',30325),('s.lesDeltaCoeff=bad;',30314),
 ('s.turbulenceModel=3;s.sstCoefficients.betaStar=bad;',30302),
 ('s.turbulenceModel=3;s.omega[0]=bad;',30302),
 ('s.turbulenceModel=3;s.sstCoefficients.betaStar=0;',30307),
 ('s.turbulenceModel=1;s.waleCw=bad;',30326)])
def test_eddy_fault_retains_output_and_exact_site(tmp_path,fault,site):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
s.turbulenceModel=2;s.smagorinskyCs=.17;s.lesDeltaCoeff=1;s.cellLength[0]=1;
s.k[0]=.1;s.omega[0]=3;s.sstWallDistance[0]=.5;s.sstCoefficients=ugkwp::defaultSstCoefficients();
s.nut[0]=123;s.sstF2[0]=456;const Real bad=std::numeric_limits<Real>::quiet_NaN();FAULT
threadIdx.x=0;computeGasEddyViscosityKernel(&s);
ck(ugkp::diagnostics::hasFatal(&d),"physical fault was hidden");
ck(d.firstFatal.sourceSiteId==SITE,"wrong producer site");
ck(s.nut[0]==123 && s.sstF2[0]==456,"invalid publication");}
'''.replace('FAULT',fault).replace('SITE',str(site)))

@pytest.mark.parametrize('fault',['sound=bad;','raw.ux=bad;','fraction=bad;','sound=0;'])
def test_characteristic_fault_is_explicit(tmp_path,fault):
    compile_probe(tmp_path,fixture()+r'''
int main(){ugkpcharacteristic::Increment raw{1,2,3,4,5};Real sound=3,fraction=.5;
const Real bad=std::numeric_limits<Real>::quiet_NaN();FAULT
ugkpcharacteristic::Failure failure;
auto result=ugkpcharacteristic::limitOneSide(raw,{1,2,3,4,5},1,0,0,1,sound,fraction,&failure);
ck(failure.failed,"characteristic fault lost");ck(!std::isfinite(result.rho),"zero-slope repair survived");}
'''.replace('FAULT',fault))

def test_mixture_characteristic_fault_records_at_caller(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
s.gasReconstruction=1;s.gasSpecies.soundSpeed[0]=std::numeric_limits<Real>::quiet_NaN();
GasPrimDevice output{123,123,123,123,123,123};Real y[2]={123,123};
ck(!reconstructGasMixtureFace(s,0,0,output,y),"caller accepted characteristic fault");
ck(d.firstFatal.sourceSiteId==30400,"caller site missing");ck(output.rho==123&&y[0]==123,"caller published failure");}
''')

@pytest.mark.parametrize('fault',['s.rhoUx[0]=bad;','s.rhoE[0]=bad;','s.gravityX=bad;'])
@pytest.mark.parametrize('mode',['false','true','true,true'])
def test_gas_gravity_fault_records_without_publication(tmp_path,fault,mode):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
const Real bad=std::numeric_limits<Real>::quiet_NaN();FAULT
const double before=s.rhoE[0];const Real beforeMy=s.rhoUy[0];applyGasGravityCell<MODE>(s,0,.5);
ck(d.firstFatal.sourceSiteId==30500,"gravity input site missing");
ck((std::isnan(before)&&std::isnan(s.rhoE[0])) || s.rhoE[0]==before,"gravity overwrote energy");
ck(s.rhoUy[0]==beforeMy,"gravity partial publication");}
'''.replace('FAULT',fault).replace('MODE',mode))

from test_native_original_success import original_native_root as original_root
from test_gas_original_success import build_run, ROOT

@pytest.mark.parametrize('model',[1,2,3])
def test_finite_eddy_outputs_match_original(tmp_path,original_root,model):
    body=fixture()+r'''
int main(){for(Real shear:{Real(0),Real(.1),Real(1),Real(10)}){
State s;initialise(s);s.turbulenceModel=MODEL;s.sstConfigured=1;s.smagorinskyCs=.17;s.waleCw=.325;s.lesDeltaCoeff=1;s.cellLength[0]=.1;
s.k[0]=.1;s.omega[0]=3;s.sstWallDistance[0]=.5;s.sstCoefficients=ugkwp::defaultSstCoefficients();
s.gradUxY[0]=shear;s.gradUyX[0]=shear/2;s.gradUzZ[0]=shear/3;threadIdx.x=0;
computeGasEddyViscosityKernel(&s);printf("%a %a\n",double(s.nut[0]),double(s.sstF2[0]));}}
'''.replace('MODEL',str(model))
    assert build_run(tmp_path,'current',ROOT,body)==build_run(tmp_path,'original',original_root,body)


def test_finite_characteristic_matches_original(tmp_path,original_root):
    body=fixture()+r'''
int main(){for(Real a:{Real(-2),Real(0),Real(.1),Real(2)}){
auto q=ugkpcharacteristic::limitOneSide({a,a/2,a/3,a/4,a/5},{.2,.1,.3,.4,.5},1,0,0,1.2,340,.5);
printf("%a %a %a %a %a\n",double(q.rho),double(q.ux),double(q.uy),double(q.uz),double(q.p));}}
'''
    assert build_run(tmp_path,'current',ROOT,body)==build_run(tmp_path,'original',original_root,body)


def test_cht_diagnosed_gravity_retains_finite_floors_and_particle_math(tmp_path):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){for(Real rho:{Real(-1),Real(0),Real(1)}){
DeviceState gas,particle;setup(gas);setup(particle);gas.rho[0]=particle.rho[0]=rho;
gas.gravityX=particle.gravityX=.3;gas.rhoUx[0]=particle.rhoUx[0]=.25;
ugkp::diagnostics::DeviceContext d{};gas.diagnostics=&d;
applyGasGravityCell<true,true>(gas,0,.2);applyGasGravityCell<true>(particle,0,.2);
ck(!ugkp::diagnostics::hasFatal(&d),"finite gravity rejected");
ck(gas.rhoE[0]==particle.rhoE[0]&&gas.rhoUx[0]==particle.rhoUx[0],"finite gravity math changed");}}
''')

@pytest.mark.parametrize('mode',['false','true','true,true'])
def test_diagnosed_gravity_preserves_state_after_prior_physical_error(tmp_path,mode):
    compile_probe(tmp_path,source_fixture()+r'''
int main(){DeviceState s;setup(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
using namespace ugkp::diagnostics;
recordFatal(&d,makeRecord(Code::NonFinite,991,991,0,EntityKind::Cell,0,-1,NAN,0,0,Quantity::Density));
s.gravityX=1;const double energy=s.rhoE[0];const Real momentum=s.rhoUx[0];
applyGasGravityCell<MODE>(s,0,.5);
ck(s.rhoE[0]==energy&&s.rhoUx[0]==momentum,"gravity published after prior physical error");
ck(d.firstFatal.sourceSiteId==991,"gravity overwrote first producer record");}
'''.replace('MODE',mode))
