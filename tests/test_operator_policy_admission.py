"""Compile policy-free entry headers without inventing missing implementations."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def compile_source(tmp_path, text, *flags):
    source=tmp_path/'entry.cpp';source.write_text(text)
    return subprocess.run(['g++','-std=c++17','-fsyntax-only','-I'+str(ROOT/'common'),*flags,str(source)],capture_output=True,text=True)


def test_policy_header_allows_existing_scalar_types_without_policy_inventory(tmp_path):
    result=compile_source(tmp_path,'#define GPU_OPERATOR_REAL int\n#define GPU_OPERATOR_TIME long long\n#define GPU_OPERATOR_THERMAL 2\n#include "GpuOperatorContract.cuh"\nint main(){GPU_OPERATOR_REAL value=3;return value;}\n')
    assert result.returncode==0,result.stderr


def test_missing_used_type_is_a_real_compiler_error(tmp_path):
    result=compile_source(tmp_path,'#include "GpuOperatorContract.cuh"\nGPU_OPERATOR_REAL value;\n')
    assert result.returncode!=0 and 'GPU_OPERATOR_REAL' in result.stderr


def test_real_definitions_are_unconditional_double(tmp_path):
    text = '#include <type_traits>\n#include "GpuPrecisionTypes.H"\nstatic_assert(std::is_same<GpuReal,double>::value,"actual storage");\nstatic_assert(std::is_same<GpuTime,double>::value && std::is_same<GpuWallEnergy,double>::value,"actual accumulation");\n'
    result = compile_source(tmp_path, text)
    assert result.returncode == 0, result.stderr
    header = (ROOT / 'common/GpuPrecisionTypes.H').read_text()
    assert 'UGKWP_GPU_REAL_BITS' not in header


def test_existing_chmt_traction_with_fixed_double_shared_storage(tmp_path):
    source=tmp_path/'traction.cpp'
    source.write_text('#include "gpu/CommonGasTraction.H"\nint main(){chmt::GasGradient gradient{};auto traction=chmt::viscousTraction(gradient,{1,0,0},1,0);return traction.x!=0;}\n')
    result=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/CHMT'),str(source),'-o',str(source.with_suffix(''))],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert subprocess.run([str(source.with_suffix(''))]).returncode==0


def test_chmt_declared_host_device_value_layouts_still_compile(tmp_path):
    source=tmp_path/'layouts.cpp'
    source.write_text('#include "core/Configuration.H"\n#include "gpu/DeviceViews.H"\n#include <type_traits>\nstatic_assert(std::is_same<chmt::Real,double>::value,"actual CHMT storage");\nint main(){chmt::GasQ q;chmt::ExchangePacket packet;chmt::Budget budget;chmt::SpeciesThermo species;chmt::PhysicsConfig config;chmt::NormalThermalView view;(void)q;(void)packet;(void)budget;(void)species;(void)config;(void)view;}\n')
    result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/CHMT'),str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_existing_nonpoisson_late_theta_specialization_initializes_theta_without_rng(tmp_path):
    source=tmp_path/'late.cpp'
    original=(ROOT/'tests/fixtures/shared_operators/uniform_pool_contract.cpp').read_text()
    extra='''
  DeviceState eager,late;
  double expected[8]={},actual[8]={};
  accumulateOnePoolParticle<false,false>(eager,0,0,1.,expected[0],expected[1],expected[2],expected[3],expected[4],expected[5],expected[6],expected[7]);
  events.clear();
  accumulateOnePoolParticle<false,true>(late,0,0,1.,actual[0],actual[1],actual[2],actual[3],actual[4],actual[5],actual[6],actual[7]);
  assert(std::memcmp(actual,expected,sizeof actual)==0);
  assert(late.pRng.value==123 && late.pStatus.value==2);
  int thetaReads=0;
  for(int event:events){assert(event!=2 && event!=80);if(event==5)++thetaReads;}
  assert(thetaReads==1 && actual[0]==2 && actual[4]==16.25);
'''
    source.write_text(original.replace('  puts("PASS uniform Poisson',extra+'  puts("PASS uniform Poisson'))
    for theta_after in (0,1):
        variant=tmp_path/f'late_{theta_after}.cpp'
        variant.write_text(source.read_text().replace('#define GPU_POOL_THETA_AFTER_REJECTION 1',f'#define GPU_POOL_THETA_AFTER_REJECTION {theta_after}'))
        exe=variant.with_suffix('')
        result=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(ROOT/'common'),str(variant),'-o',str(exe)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        assert run.returncode==0,run.stdout+run.stderr


def test_actual_zero_count_species_types_compile_without_positive_count_policy(tmp_path):
    source=tmp_path/'zero_species.cpp'
    source.write_text('''#include "gasTransport/GasStateView.H"
#include "core/Types.H"
int main(){
 ugkwp::SpeciesThermoView<double,0> thermo;
 ugkwp::GasMechanismView<double,0> mechanism;
 ugkwp::ClosedReactorInput<double,0> reactor;
 ugkwp::GasSpeciesState<double,0> state;
 chmt::GasQ gas;gas.mass=2;auto total=gas+gas;
 (void)thermo;(void)mechanism;(void)reactor;(void)state;(void)gas;
 return ugkwp::compiledGasSpecies==0 && total.mass==4?0:2;
}''')
    result=subprocess.run(['g++','-std=c++17','-O2','-DUGKWP_GAS_SPECIES=0','-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/CHMT'),str(source),'-o',str(source.with_suffix(''))],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert subprocess.run([str(source.with_suffix(''))]).returncode==0


def test_negative_sized_species_arrays_remain_actual_compiler_errors(tmp_path):
    source=tmp_path/'negative_species.cpp'
    source.write_text('#include "core/Types.H"\nint main(){chmt::GasQ gas;(void)gas;}\n')
    result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-DUGKWP_GAS_SPECIES=-1','-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/CHMT'),str(source)],capture_output=True,text=True)
    assert result.returncode!=0 and 'array' in result.stderr
    assert 'static assertion failed' not in result.stderr
