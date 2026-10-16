"""Physical and residual classification at actual gas cell producers."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def test_nonfinite_species_flux_not_published(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.gasSpecies.flux[0]=std::numeric_limits<Real>::quiet_NaN();const auto before=s.gasSpecies.rho[0];
 threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01));
 ck(ugkp::diagnostics::hasFatal(&d),"species flux NaN not fatal");ck(s.gasSpecies.rho[0]==before,"invalid species published");
}
''')


def test_nonfinite_recovery_stops_before_eos(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.gasSpecies.rho[0]=std::numeric_limits<Real>::quiet_NaN();const auto before=s.Tgas[0];recoverGasPrimitiveCell(s,0);
 ck(ugkp::diagnostics::hasFatal(&d),"species NaN not fatal");ck(s.Tgas[0]==before,"invalid primitive published");
}
''')


def test_finite_species_flux_residual_warns_and_continues(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.gasSpecies.flux[0]=.01;const auto before=s.gasSpecies.rho[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,Real(.01),true);
 ck(!ugkp::diagnostics::hasFatal(&d),"finite conservation residual fatal");ck(d.warnings[0].count>0,"residual unreported");ck(s.gasSpecies.rho[0]!=before,"residual rejected arithmetic");
}
''')


def test_periodic_nonfinite_species_does_not_publish(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;
 s.gasSpecies.flux[0]=std::numeric_limits<Real>::quiet_NaN();s.gasSpecies.flux[1]=7;
 threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
 ck(ugkp::diagnostics::hasFatal(&d),"periodic NaN not fatal");ck(s.gasSpecies.flux[1]==7,"invalid periodic counterpart published");
}
''')


def test_periodic_finite_closure_residual_warns(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;
 s.gasSpecies.flux[0]=1;s.gasSpecies.flux[1]=-1;s.gasSpecies.flux[2]=2;s.gasSpecies.flux[3]=-2;
 threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s,true);
 ck(!ugkp::diagnostics::hasFatal(&d),"finite periodic residual fatal");ck(d.warnings[0].count>0,"periodic residual unreported");
 ck(s.gasSpecies.flux[0]==1&&s.gasSpecies.flux[2]==2,"periodic residual changed values");
}
''')
