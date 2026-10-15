"""Production gas kernels with independently stored FP64 energy, run on the CPU.

The optional CUDA compile/runtime checks are separate; these tests claim neither.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture


def energy_fixture():
    # The RED baseline used hidden double pointers to execute the old numerical
    # bodies; the final fixture exercises the typed public interface itself.
    return fixture().replace(
        'GasStateView<Real,Time,ugkwp::SstCoefficients>',
        'GasStateView<Real,Time,ugkwp::SstCoefficients,double>',
    ).replace('s.rhoE=new Real[8]{};', 's.rhoE=new double[8]{};').replace(
        's.rhoENext=new Real[8]{};', 's.rhoENext=new double[8]{};'
    ) + r'''
void closeEnergy(double actual,long double expected,const char* message,double tolerance=2.4e-7){
 if(std::abs((long double)actual-expected)>tolerance){std::fprintf(stderr,"%s actual=%.17g expected=%.21Lg error=%.9Lg\n",message,actual,expected,(long double)actual-expected);std::abort();}
}
'''


def test_actual_flux_signed_increments_survive_energy_storage_and_eos(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){
 for(double increment:{.02,.03,.05,-.02,-.03,-.05,-.0001}){
  State s;initialise(s);s.rhoE[0]=1e9;s.V[0]=Real(.7);s.gasPhiRhoE[0]=Real(-increment*.7);
  const double old=s.rhoE[0],dt=1.;const long double actualIncrement=-(long double)s.gasPhiRhoE[0]*dt/(long double)s.V[0];
  applyGasFluxDivergenceByCellKernel(&s,dt);
  closeEnergy(s.rhoE[0],(long double)old+actualIncrement,"actual final-flux increment lost");
  if(std::abs(increment)>=.02)ck(std::abs((s.rhoE[0]-old)-increment)/std::abs(increment)<.002,"physical increment relative error");
  const double before=s.rhoE[0];recoverGasPrimitivesKernel(&s);
  ck(std::memcmp(&before,s.rhoE,sizeof before)==0,"EOS destroyed authoritative energy bits");
 }
 State s;initialise(s);s.rhoE[0]=1e9;s.gasPhiRhoE[0]=Real(-.03);long double expected=1e9L;
 for(int i=0;i<10000;++i){applyGasFluxDivergenceByCellKernel(&s,1.);recoverGasPrimitivesKernel(&s);expected-=(long double)s.gasPhiRhoE[0];}
 closeEnergy(s.rhoE[0],expected,"long sequence energy drift",.0013);
}
''')


def test_energy_flux_accumulator_is_independent_from_real(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.nCells=1;s.cellPlaneCount[0]=3;s.cellFaceId[2]=2;s.faceOwner[2]=0;
s.rhoE[0]=1e9;s.gasPhiRhoE[0]=Real(1e8);s.gasPhiRhoE[1]=Real(.03);s.gasPhiRhoE[2]=Real(-1e8);
applyGasFluxDivergenceByCellKernel(&s,1.);closeEnergy(s.rhoE[0],1e9L-(long double)Real(.03),"signed face sum lost small flux");}
''')


def test_rk_weights_and_zero_flux_preserve_authoritative_energy(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rhoE[0]=1000000000.03;saveGasConservativeStateKernel(&s);
for(int i=0;i<1000;++i){blendGasConservativeStateKernel(&s,1./3.,2./3.);ck(s.rhoE[0]==s.rhoENext[0],"RK zero-flux injects energy");}
s.rhoE[0]=s.rhoENext[0]+.03;const long double expected=(long double)s.rhoENext[0]+((long double)s.rhoE[0]-s.rhoENext[0])*2.L/3.L;
blendGasConservativeStateKernel(&s,1./3.,2./3.);closeEnergy(s.rhoE[0],expected,"RK weights were rounded through Real");}
''')


def test_ale_energy_uses_unrounded_volume_ratio_and_scale(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={Real(.7),Real(1.3)},newV[2]={Real(.8),Real(1.2)},sweep[3]={Real(.1),0,0};moving(s,oldV,newV,sweep,Real(.125));
s.rhoE[0]=1e9+.03;s.gasPhiRhoE[0]=Real(-.4);
const long double expected=((long double)s.rhoE[0]*(long double)oldV[0]-.125L*(long double)s.gasPhiRhoE[0])/(long double)newV[0];
applyGasFluxDivergenceByCellKernel(&s,.125);closeEnergy(s.rhoE[0],expected,"ALE energy used rounded Real geometry ratio");}
''')


def test_energy_view_binding_and_lifetime_preserve_pointer_types(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rhoE[0]=1e9+.03;auto view=ugkwp::makeGasStateView(s);
static_assert(std::is_same<decltype(view.rhoE),double*>::value,"energy type lost");
static_assert(std::is_same<decltype(view.rho),Real*>::value,"mass type changed");
static_assert(std::is_trivially_copyable<decltype(view)>::value,"view not POD");
ck(view.rhoE==s.rhoE&&view.rhoENext==s.rhoENext,"binder did not preserve authority");
saveGasConservativeStateKernel(&view);ck(s.rhoENext[0]==s.rhoE[0],"RK save narrowed energy");
auto copied=view;double replacement[2]={1e9+.05,0};s.rhoE=replacement;auto rebound=ugkwp::makeGasStateView(s);
ck(copied.rhoE!=rebound.rhoE&&rebound.rhoE==replacement,"copy/rebind lifetime");
const float oldCheckpoint=float(1e9+.03);rebound.rhoE[0]=oldCheckpoint;ck(rebound.rhoE[0]==double(oldCheckpoint),"old float promotion changes bits");}
''')


def mixture_fixture(ns):
    return energy_fixture().replace('GasSpeciesState<Real,2>','GasSpeciesState<Real,'+str(ns)+'>')+r'''
#include "operators/advanceGasChemistryKernel.cuh"
constexpr int N=SPECIES;
struct MixtureFixture{
 State state;ugkwp::SpeciesThermoData<Real> data[N];Real coeff[3*N]={},partial[2*N]={},sound[2]={},cp[2]={},rgas[2]={};
 ugkwp::ChemistryStatus chemistryStatus[2];
 ugkwp::GasReactionData<Real> reactions[1];ugkwp::GasStoichTerm<Real> reactants[1],products[1];Real basis[N]={};
 MixtureFixture(){initialise(state);auto&g=state.gasSpecies;g.mode=ugkwp::GasMode::MixtureFrozen;
  for(int k=0;k<N;++k){data[k].molarMass=Real(.028);data[k].coefficientOffset=3*k;data[k].minTemperature=100;data[k].maxTemperature=3000;data[k].referencePressure=101325;coeff[3*k]=1000;coeff[3*k+2]=-1e6;}
  g.thermo.species=data;g.thermo.coefficients=coeff;g.thermo.coefficientCount=3*N;g.rho=partial;g.soundSpeed=sound;g.heatCapacity=cp;g.gasConstant=rgas;g.chemistryStatus=chemistryStatus;
  partial[0]=state.rho[0];partial[1]=state.rho[1];g.mechanism.referencePressure=101325;
  g.thermoControls.relativeEnergyTolerance=Real(1e-10);g.thermoControls.relativeTemperatureTolerance=Real(1e-10);
  reactions[0].reactantCount=reactions[0].productCount=1;reactions[0].highRate.preExponential=100;reactions[0].highRate.activationEnergy=8000;reactants[0].species=0;products[0].species=1;reactants[0].coefficient=products[0].coefficient=1;basis[0]=-1;basis[1]=1;
  g.mechanism.reactions=reactions;g.mechanism.reactionCount=1;g.mechanism.reactants=reactants;g.mechanism.products=products;g.mechanism.reactantTermCount=g.mechanism.productTermCount=1;g.mechanism.stoichiometricBasis=basis;g.mechanism.independentRank=1;
  state.rhoUx[0]=Real(2e7);const double density=state.rho[0],mx=state.rhoUx[0];
  const double internal=density*ugkwp::speciesE(0,700.,g.thermo);state.rhoE[0]=mx*mx/(2*density)+internal;
 }
};
'''.replace('SPECIES',str(ns))


@pytest.mark.parametrize('ns',[2,10])
def test_mixed_table_eos_reads_authoritative_high_kinetic_energy(tmp_path,ns):
    compile_probe(tmp_path,mixture_fixture(ns)+r'''
int main(){MixtureFixture f;auto&s=f.state;const double before=s.rhoE[0];recoverGasPrimitivesKernel(&s);
ck(s.gasSpecies.cellStatus[0]==0,"mixed-table high-kinetic EOS rejected valid authoritative energy");
ck(std::abs(double(s.Tgas[0])-700.)<.005,"mixed-table EOS narrowed internal energy");
ck(s.rhoE[0]==before,"mixed EOS rewrote energy");}
''')


@pytest.mark.parametrize('ns',[2,10])
def test_chemistry_reads_double_energy_and_explicit_stage_volume(tmp_path,ns):
    compile_probe(tmp_path,mixture_fixture(ns)+r'''
int main(){MixtureFixture f;auto&s=f.state;s.gasSpecies.mode=ugkwp::GasMode::MixtureChemistry;
s.gasSpecies.chemistryControls.relativeConservationTolerance=Real(1e-5);
s.gasSpecies.chemistryControls.thermo.relativeEnergyTolerance=Real(1e-5);s.gasSpecies.chemistryControls.thermo.relativeTemperatureTolerance=Real(1e-5);
const double before=s.rhoE[0];ck(bool(ugkwp::advanceGasChemistryCell(s,0,0.,Real(.7))),"chemistry input narrowed authoritative internal energy");
ck(s.rhoE[0]==before,"chemistry added formation energy twice");ck(s.gasSpecies.rho[0]==s.rho[0],"zero chemistry altered species");
ck(bool(ugkwp::advanceGasChemistryCell(s,0,.01,Real(.3))),"active chemistry rejected");
const double expected=double(s.rho[0])*std::exp(-100.*std::exp(-8000./(ugkwp::universalGasConstant<double>()*700.))*.01);
ck(std::abs(double(s.gasSpecies.rho[0])-expected)<.002,"reaction rate responded to truncated energy");
ck(s.rhoE[0]==before,"active chemistry rewrote formation-inclusive energy");
}
''')


def test_mixed_energy_flux_cache_conversion_is_explicit(tmp_path):
    from pathlib import Path
    import subprocess
    from test_mixture_state import ROOT,HERE
    probe=(HERE/'gas_state_probe.cpp').read_text()
    source=tmp_path/'flux.cpp'
    source.write_text(probe[:probe.index('void hashBytes')]+energy_fixture()+r'''
int main(){State s;initialise(s);flux(s,.001);}
''')
    result=subprocess.run(['g++','-std=c++17','-Werror=narrowing','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(tmp_path/'flux')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_mixed_eos_rejects_unrepresentable_primitive_without_publishing(tmp_path):
    compile_probe(tmp_path,mixture_fixture(2)+r'''
int main(){MixtureFixture f;auto&s=f.state;for(int k=0;k<2;++k){f.data[k].molarMass=Real(1e30);f.coeff[3*k+2]=0;}
s.rho[0]=std::numeric_limits<Real>::denorm_min();s.rhoUx[0]=0;f.partial[0]=s.rho[0];s.rhoE[0]=double(s.rho[0])*ugkwp::speciesE(0,700.,s.gasSpecies.thermo);
const double oldE=s.rhoE[0];recoverGasPrimitivesKernel(&s);
ck(s.gasSpecies.cellStatus[0]==0&&s.p[0]==0,"finite pressure underflow refused");
ck(s.rhoE[0]==oldE&&std::isfinite(s.Tgas[0]),"underflow corrupted Energy authority");}
''')


@pytest.mark.parametrize('strict',[False,True])
def test_mixed_ale_numeric_candidates_and_rk_energy(tmp_path,strict):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rhoE[0]=1e9+.03;saveGasConservativeStateKernel(&s);
const double before=s.rhoE[0],saved=s.rhoENext[0];Real oldV[2]={1,1},newV[2]={Real(1.1),Real(.9)},sweeps[3]={Real(.1),0,0};moving(s,oldV,newV,sweeps,Real(.125));
applyGasFluxDivergenceByCellKernel(&s,.25,STRICT);ck(s.gasSpecies.cellStatus[0]==0,"finite interval difference refused");
const double updated=s.rhoE[0];ck(updated==before/double(newV[0])&&s.rhoENext[0]==saved,"actual interval geometry or RK save lost");
const Real live[4]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0]};
s.gasSpecies.cellStatus[0]=0;s.gasPhiRhoE[0]=std::numeric_limits<Real>::quiet_NaN();applyGasFluxDivergenceByCellKernel(&s,.125,STRICT);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::NonFiniteState)&&s.rhoENext[0]==saved,"numeric failure or saved energy lost");
const Real candidate[4]={s.rho[0],s.rhoUx[0],s.rhoUy[0],s.rhoUz[0]};
if(STRICT)ck(s.rhoE[0]==updated&&std::memcmp(live,candidate,sizeof live)==0,"strict candidate published physical fields");
else{ck(std::isnan(s.rhoE[0]),"public energy arithmetic was not evaluated");for(int k=0;k<4;++k)ck(candidate[k]==live[k]*(oldV[0]/newV[0]),"public bulk arithmetic incomplete");}
// RK save/restore is exercised directly, without claiming a CHT transaction layer.
s.gasSpecies.cellStatus[0]=0;s.rhoE[0]=updated+.05;blendGasConservativeStateKernel(&s,1.,0.);ck(s.rhoE[0]==saved,"RK restore narrowed accepted bits");}
'''.replace('STRICT','true' if strict else 'false'))


def test_double_production_kernel_preserves_small_energy_increments(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){for(double change:{.02,.03,.05}){State s;initialise(s);s.rhoE[0]=Real(1e9);s.gasPhiRhoE[0]=Real(-change);
const Real before=s.rhoE[0];applyGasFluxDivergenceByCellKernel(&s,1.);ck(s.rhoE[0]>before,"double energy increment vanished");ck(s.rhoE[0]==double(static_cast<long double>(before)+static_cast<long double>(change)),"energy increment differs from independently rounded sum");}}
''')


def test_mixed_legacy_recovery_preserves_existing_invalid_velocity_repair(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rho[0]=s.rhoMin;s.rhoUx[0]=std::numeric_limits<Real>::max();s.rhoE[0]=0;
recoverGasPrimitivesKernel(&s);ck(s.rhoUx[0]==0,"legacy invalid velocity was not sanitized");
ck(s.rhoE[0]>0&&s.rhoE[0]<1,"energy repair used pre-sanitized overflow velocity");}
''')


def test_legacy_high_kinetic_recovery_preserves_small_internal_increment(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rhoUx[0]=Real(2e7);const double kinetic=double(s.rhoUx[0])*double(s.rhoUx[0])/(2*double(s.rho[0]));s.rhoE[0]=kinetic+1000000.03;
const double before=s.rhoE[0];recoverGasPrimitivesKernel(&s);ck(s.rhoE[0]==before,"legacy high kinetic recovery rewrote valid energy");

s.rhoE[0]=kinetic-1;recoverGasPrimitivesKernel(&s);ck(s.rhoE[0]>=kinetic,"existing internal energy floor was removed");}
''')


def test_double_legacy_publishes_large_finite_pressure_without_changing_energy(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rhoE[0]=1e40;const double before=s.rhoE[0];recoverGasPrimitivesKernel(&s);
ck(s.gasSpecies.cellStatus[0]==0,"finite double primitive was rejected");
ck(s.rhoE[0]==before&&std::isfinite(s.p[0])&&s.p[0]>1e39&&std::isfinite(s.Tgas[0]),"large finite primitive or conserved authority corrupted");}
''')


def test_mixed_ale_rejects_pressure_underflow_before_primitive_publication(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={1,1},sweeps[3]={};moving(s,oldV,newV,sweeps,Real(.125));
s.rho[0]=s.rhoMin=std::numeric_limits<Real>::denorm_min();s.rhoE[0]=double(s.rho[0])*double(s.Rgas)*1e-5/(double(s.gammaGas)-1);
const double oldE=s.rhoE[0];recoverGasPrimitivesKernel(&s);
ck(s.gasSpecies.cellStatus[0]==0&&s.p[0]==0,"mixed ALE refused finite underflow");ck(s.rhoE[0]==oldE&&std::isfinite(s.Tgas[0]),"mixed ALE underflow changed Energy authority");}
''')


def test_repeated_high_kinetic_recovery_preserves_all_valid_conserved_bits(tmp_path):
    compile_probe(tmp_path,energy_fixture()+r'''
int main(){State s;initialise(s);s.rho[0]=Real(1.1);s.rhoUx[0]=Real(19999000.875);
const double density=s.rho[0],mx=s.rhoUx[0],kinetic=.5*density*(mx/density)*(mx/density);s.rhoE[0]=kinetic+1e6;
const Real rhoBefore=s.rho[0],mxBefore=s.rhoUx[0];const double energyBefore=s.rhoE[0];
const double expectedT=(energyBefore-kinetic)*(double(s.gammaGas)-1)/(density*double(s.Rgas));
for(int repeat=0;repeat<10;++repeat){recoverGasPrimitivesKernel(&s);
 ck(std::memcmp(&rhoBefore,s.rho,sizeof rhoBefore)==0,"valid density changed on recovery");
 ck(std::memcmp(&mxBefore,s.rhoUx,sizeof mxBefore)==0,"valid momentum changed on recovery");
 ck(std::memcmp(&energyBefore,s.rhoE,sizeof energyBefore)==0,"zero-change recovery invented gas energy");
 ck(std::abs(double(s.Tgas[0])-expectedT)<.001,"high-kinetic recovered temperature changed");
 ck(s.Tgas[0]>1000,"valid high-kinetic state fell to lower bound");}}
''')


def test_production_cht_owner_has_double_live_and_rk_authority():
    from pathlib import Path
    owner=((SOURCE_ROOT / 'applications/CHT/gpu/GpuResidentStrict.cu')).read_text()
    assert "double* rhoE = nullptr" in owner
    assert "double* rhoENext = nullptr" in owner
