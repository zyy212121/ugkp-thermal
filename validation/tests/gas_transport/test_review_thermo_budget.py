import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize("native", [False, True])
def test_usable_caloric_iteration_limit_is_warning(tmp_path, native):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
s.gasSpecies.thermoControls.maximumIterations=0;s.gasSpecies.thermo.diagnostics=&diagnostics;
s.Tgas[0]=400;
Real mass[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[2]};Real candidate=400;bool available=false;
auto result=ugkwp::invertMixtureEnergy(mass,s.rhoE[0],s.gasSpecies.thermo,s.gasSpecies.thermoControls,candidate,&available);
ck(result==ugkwp::ThermoStatus::Success && available && candidate==400,"finite numerical limit must publish its available estimate");
diagnostics={};threadIdx.x=0;recoverGasPrimitivesKernel(&s,NATIVE);
ck(!diagnostics.fatalClaim,"caloric numerical residual became fatal");
ck(s.gasSpecies.cellStatus[0]==0,"caloric numerical residual rejected cell");
ck(std::isfinite(s.Tgas[0]) && s.Tgas[0]>0,"usable temperature not published");
ck(diagnostics.warnings[1].count || diagnostics.warnings[2].count,"iteration-limit warning missing");
}'''.replace('NATIVE', 'true' if native else 'false'))
