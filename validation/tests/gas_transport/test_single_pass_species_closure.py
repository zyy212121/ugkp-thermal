"""Closure uses one predetermined largest carrier, never retries another species."""
from test_mixture_transport import fixture
from test_mixture_state import compile_probe


def test_rk_roundoff_uses_largest_carrier_once(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;
const Real a=.03027317344064806,b=.9811327243011391;
s.gasSpecies.rho[0]=a;s.gasSpecies.rho[2]=1-a;saveGasConservativeStateKernel(&s);
s.gasSpecies.rho[0]=b;s.gasSpecies.rho[2]=1-b;
const Real minor=.25*(1-a)+.75*(1-b);
blendGasConservativeStateKernel(&s,Real(.25),Real(.75));
ck(s.gasSpecies.rho[2]==minor,"closure changed the minor carrier");
ck(s.gasSpecies.rho[0]==1-minor,"largest carrier not closed once");
ck(s.rho[0]==1,"closure changed bulk mass");
}
''')


def test_unrepairable_finite_species_residual_warns_and_retains_inventory(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
s.gasSpecies.rho[0]=.3;s.gasSpecies.rho[2]=.3;saveGasConservativeStateKernel(&s);
s.gasSpecies.rho[0]=.1;s.gasSpecies.rho[2]=.1;
blendGasConservativeStateKernel(&s,Real(.5),Real(.5));
ck(s.gasSpecies.rho[0]==.2&&s.gasSpecies.rho[2]==.2,"residual changed physical species inventory");
ck(s.rho[0]==1&&!diagnostics.fatalClaim&&!s.gasSpecies.cellStatus[0],"finite closure residual stopped cell");
ck(diagnostics.warnings[0].count>0,"finite residual was silent");
}
''')


def test_species_audit_sum_overflow_is_warning_only(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
const Real huge=1.2e308;
s.gasSpecies.rho[0]=huge;s.gasSpecies.rho[2]=huge;saveGasConservativeStateKernel(&s);
blendGasConservativeStateKernel(&s,Real(.5),Real(.5));
ck(s.gasSpecies.rho[0]==huge&&s.gasSpecies.rho[2]==huge,"audit overflow altered finite inventories");
ck(!diagnostics.fatalClaim&&diagnostics.warnings[0].count>0,"audit sum overflow must warn");
s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;
s.gasSpecies.flux[0]=huge;s.gasSpecies.flux[1]=0;s.gasSpecies.flux[2]=huge;s.gasSpecies.flux[3]=0;
enforcePeriodicGasFluxAntisymmetryKernel(&s);
ck(!diagnostics.fatalClaim,"periodic audit overflow fatal");
ck(s.gasSpecies.flux[0]==huge*.5&&s.gasSpecies.flux[2]==huge*.5,"finite flux lost on audit overflow");
}
''')


def test_divergence_flux_audit_overflow_keeps_finite_update(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
s.gasSpecies.flux[0]=1.2e308;s.gasSpecies.flux[3]=-1.2e308;
const Real expected0=s.gasSpecies.rho[0]-1e-310*s.gasSpecies.flux[0];
const Real expected1=s.gasSpecies.rho[2]-1e-310*s.gasSpecies.flux[3];
applyGasFluxDivergenceByCellKernel(&s,1e-310);
ck(!diagnostics.fatalClaim&&!s.gasSpecies.cellStatus[0],"flux magnitude audit overflow stopped finite divergence");
ck(diagnostics.warnings[0].count>0,"flux audit overflow silent");
ck(s.gasSpecies.rho[0]==expected0&&s.gasSpecies.rho[2]==expected1,"actual finite divergence discarded");
}
''')


def test_large_finite_legacy_and_sst_values_are_not_replaced_by_floor(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);threadIdx.x=0;
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
s.rho[0]=1e301;s.rhoE[0]=1e302;s.Rgas=1;
recoverGasPrimitiveCell(s,0);
ck(!diagnostics.fatalClaim&&s.rho[0]==1e301&&s.rhoE[0]==1e302,"finite legacy inventory replaced by magnitude sentinel");
s.rho[0]=1;s.sstConfigured=1;s.sstWallTreatment=3;s.sstWallKappa=.41;s.sstWallE=9.8;s.turbulentPrandtl=.9;s.k[0]=1e301;s.omega[0]=1e301;
initialiseSstConservativeStateKernel(&s);
ck(!diagnostics.fatalClaim&&s.rhoK[0]==1e301&&s.rhoOmega[0]==1e301,"finite SST initialization replaced by floor");
recoverSstPrimitiveCell(s,0);
ck(!diagnostics.fatalClaim&&s.k[0]==1e301&&s.omega[0]==1e301,"finite SST recovery replaced by floor");
}
''')
