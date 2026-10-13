"""Valid FP64 authority must survive legacy EOS recovery without one-ULP repair."""
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture


def test_valid_double_energy_nextafter_bits_survive_recovery(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.rho[0]=2;s.rhoUx[0]=.00048828125;s.rhoE[0]=std::nextafter(1e9,2e9);
const double energy=s.rhoE[0],momentum=s.rhoUx[0],density=s.rho[0];
const double kinetic=.5*density*(momentum/density)*(momentum/density);
const double expectedP=(s.gammaGas-1)*(energy-kinetic),expectedT=expectedP/(density*s.Rgas);
recoverGasPrimitivesKernel(&s);
ck(std::memcmp(&energy,s.rhoE,sizeof energy)==0,"valid FP64 EOS recovery lost one authoritative ULP");
ck(s.rho[0]==density&&s.rhoUx[0]==momentum,"identity recovery changed mass/momentum");
ck(s.p[0]==expectedP&&s.Tgas[0]==expectedT,"physical EOS outputs changed calculation");}
''')


def test_double_recovery_retains_existing_invalid_and_low_internal_repairs(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(double bad:{-1.,std::numeric_limits<double>::quiet_NaN()}){State s;initialise(s);s.rho[0]=2;s.rhoUx[0]=6;s.TgasMin=300;s.rhoE[0]=bad;
const double kinetic=9,minimum=std::max(double(s.rhoMin),2.*s.Rgas*s.TgasMin/(s.gammaGas-1));recoverGasPrimitivesKernel(&s);
ck(s.rhoE[0]==kinetic+minimum,"existing FP64 energy lower-bound repair changed");ck(s.rhoUx[0]==6&&s.Ux[0]==3,"valid momentum changed during floor repair");ck(std::abs(s.Tgas[0]-300.)<1e-12,"floor-repaired temperature incorrect");}}
''')


def test_zero_flux_double_rk_thirds_preserve_nextafter_bits(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.rhoE[0]=std::nextafter(std::nextafter(1e9,2e9),2e9);saveGasConservativeStateKernel(&s);
const double energy=s.rhoE[0];for(int stage=0;stage<1000;++stage)blendGasConservativeStateKernel(&s,1./3.,2./3.);
ck(std::memcmp(&energy,s.rhoE,sizeof energy)==0,"zero-flux double RK thirds lost authoritative ULP");
// This guard applies only to an affine identity, not arbitrary weight sums.
blendGasConservativeStateKernel(&s,.5,.25);ck(s.rhoE[0]==.75*energy,"non-unit weight sum was incorrectly treated as identity");}
''')


def test_repeated_double_eos_preserves_valid_high_kinetic_momentum(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.rho[0]=1.1;s.rhoUx[0]=19999000.875;const double rho=s.rho[0],mx=s.rhoUx[0],u=mx/rho;
s.rhoE[0]=.5*rho*u*u+1e6;const double energy=s.rhoE[0],expectedT=1e6*(s.gammaGas-1)/(rho*s.Rgas);
for(int i=0;i<10;++i){recoverGasPrimitivesKernel(&s);ck(std::memcmp(s.rho,&rho,sizeof rho)==0&&std::memcmp(s.rhoUx,&mx,sizeof mx)==0&&std::memcmp(s.rhoE,&energy,sizeof energy)==0,"valid double conserved state changed on repeated EOS");ck(std::abs(s.Tgas[0]-expectedT)<.001,"repeated double EOS changed physical temperature");}}
''')


def test_double_eos_keeps_invalid_density_and_momentum_repair(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.rhoMin=.5;s.rho[0]=.25;s.rhoUx[0]=3;s.rhoUy[0]=std::numeric_limits<double>::infinity();s.rhoE[0]=-1;s.TgasMin=300;
recoverGasPrimitivesKernel(&s);ck(s.rho[0]==.5&&s.rhoUx[0]==3&&s.rhoUy[0]==0,"existing density or invalid-momentum repair changed");
ck(s.Ux[0]==6&&s.Uy[0]==0,"repaired primitive velocity changed");ck(std::abs(s.Tgas[0]-300.)<1e-12,"repaired temperature below floor");
const double minimum=.5*s.Rgas*s.TgasMin/(s.gammaGas-1);ck(s.rhoE[0]==9+minimum,"existing repaired energy changed");}
''')
