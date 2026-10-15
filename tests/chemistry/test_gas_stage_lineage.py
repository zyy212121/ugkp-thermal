"""Exact original-success numerical lineage, using immutable production headers.

The previous finite-candidate/compensated-time expectations encoded T03's
rejected algorithm. These tests compare actual original and current arithmetic,
including successful zero-effect subnormal steps and retryable failure counters.
Independent Cantera/analytic/conservation assertions remain in their own tests.
"""
from pathlib import Path
import io
import subprocess
import tarfile

import pytest
from test_gas_rates import SOURCE as RATE_SOURCE
from test_h2o2_reactor import SOURCE as H2O2_SOURCE, REFERENCE
from test_gas_chemistry_adapter import SOURCE as ADAPTER_SOURCE

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = 'd6110c02745a8ad726208a426c4bb3a28233c1fd'


LEGACY_CALL_ADAPTER = r'''
#include "common/chemistry/LocalStiffIntegrator.H"
template<class R,int N,class TableReal>
ugkwp::ChemistryStatus testOriginalReactor(const ugkwp::ClosedReactorInput<R,N>& b,R dt,
 const ugkwp::SpeciesThermoView<TableReal,N>& t,const ugkwp::GasMechanismView<TableReal,N>& m,
 const ugkwp::ChemistryControls<R>& c,ugkwp::ClosedReactorResult<R,N>& out){
 ugkwp::ChemistryAudit<R,N> legacyOutput;
 return ugkwp::advanceClosedReactor(b,dt,t,m,c,out,legacyOutput);
}
'''


@pytest.fixture(scope='module')
def original_root(tmp_path_factory):
    folder = tmp_path_factory.mktemp('original-chemistry-headers')
    archive = subprocess.run(['git', 'archive', ORIGINAL, 'common'], cwd=ROOT,
                             capture_output=True)
    assert archive.returncode == 0, archive.stderr.decode()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as source:
        source.extractall(folder, filter='data')
    return folder


def build(folder, label, root, source):
    src = folder / (label + '.cpp')
    exe = folder / label
    if root != ROOT:


        source = source.replace('advanceClosedReactor(', 'testOriginalReactor(')
        source = LEGACY_CALL_ADAPTER + source
    src.write_text(source)
    command = ['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra',
               '-Werror', '-I' + str(root), str(src), '-o', str(exe)]
    built = subprocess.run(command, capture_output=True, text=True)
    (folder / (label + '_build.log')).write_text(built.stdout + built.stderr)
    assert built.returncode == 0, built.stderr
    return exe


def run(exe, inputs=''):
    result = subprocess.run([str(exe)], input=inputs, capture_output=True,
                            text=True, timeout=180)
    exe.with_suffix('.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    return result.stdout


REACTOR = RATE_SOURCE.split('int main(){')[0] + r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <cstdio>
void run(const char* name,double rate,double interval,double maximumStep,int budget=10000){
 Model a;a.rx[0].highRate.preExponential=rate;
 ClosedReactorInput<double,2> b;b.speciesMass[0]=.028;b.totalMass=.028;b.volume=.001;b.internalEnergy=mixtureEnergy(b.speciesMass,700.,a.t);
 ChemistryControls<double> c;c.relativeTolerance=1e-7;c.absoluteMassFractionTolerance=1e-13;c.absoluteTemperatureTolerance=1e-6;c.maximumStep=maximumStep;c.maximumSteps=budget;
 PRECISION_CONTROLS
 ClosedReactorResult<double,2> result;result.speciesMass[0]=result.speciesMass[1]=result.temperature=-777;
 const auto status=ENTRY(b,interval,a.t,a.m,c,result);
 std::printf("%s %d %d %d %d %d %a %a %a\n",name,int(status.code),int(status.thermoCode),status.reaction,status.attemptedSteps,status.rejectedSteps,
 double(result.speciesMass[0]),double(result.speciesMass[1]),double(result.temperature));
}
int main(){
 run("ordinary",2,.25,0);run("fine",2,.25,.0001);run("stiff",1e8,.001,0);
 for(int n:{1,2,3,7,17})run("subnormal",1e20,n*std::numeric_limits<double>::denorm_min(),0);
 EXTRA_CASES
}
'''


@pytest.mark.parametrize('retry', [False, True], ids=['successful-public', 'retry-owner'])
def test_exact_reactor_state_and_work_lineage(tmp_path, original_root, retry):
    source = REACTOR.replace('PRECISION_CONTROLS', '')
    source = source.replace('EXTRA_CASES', 'run("budget",2,.25,0,1);' if retry else '')
    source = '#include <initializer_list>\n' + source
    before = build(tmp_path, 'original', original_root, source.replace('ENTRY', 'advanceClosedReactor'))
    after = build(tmp_path, 'corrected', ROOT, source.replace('ENTRY', 'advanceClosedReactorTrial' if retry else 'advanceClosedReactor'))
    expected, actual = run(before), run(after)
    assert actual == expected
    if not retry:
        assert all(line.split()[1] == '0' for line in expected.splitlines())


def test_exact_h2o2_histories_and_refinement(tmp_path, original_root):

    source = H2O2_SOURCE
    original = build(tmp_path, 'original_h2o2', original_root, source)
    corrected = build(tmp_path, 'corrected_h2o2', ROOT, source)
    for case in REFERENCE['cases']:
        for tolerance in [1e-6, 1e-7]:
            initial = case['states'][0]
            inputs = [case['volume'], *initial['speciesMass'], initial['temperature'],
                      tolerance, len(case['output_times']), *case['output_times']]
            before = run(original, ' '.join(map(str, inputs)))
            after = run(corrected, ' '.join(map(str, inputs)))
            (tmp_path / (case['name'] + '_' + str(tolerance) + '_original.log')).write_text(before)
            (tmp_path / (case['name'] + '_' + str(tolerance) + '_corrected.log')).write_text(after)
            assert after == before, (case['name'], tolerance)


def test_exact_density_adapter_publication(tmp_path, original_root):
    source = ADAPTER_SOURCE.split('int main(){')[0].replace('TEST_REAL', 'double') + r'''
#include <cstdio>
int main(){
 for(double interval:{0.,.1000000000001,.25})for(Real volume:{Real(.3),Real(.5),Real(2)}){
  Fixture f;auto status=advanceGasChemistryCell(f.state,0,interval,volume);assert(status);
  std::printf("status %d %d %d %d %d %d\n",int(status.code),int(status.thermoCode),status.reaction,status.attemptedSteps,status.rejectedSteps,f.cellStatus[0]);
  for(Real x:f.partial)std::printf("%a ",double(x));
  for(Real x:f.energy)std::printf("%a ",double(x));
  for(Real x:f.rho)std::printf("%a ",double(x));
  for(Real x:f.mx)std::printf("%a ",double(x));
  for(Real x:f.my)std::printf("%a ",double(x));
  for(Real x:f.mz)std::printf("%a ",double(x));
  for(Real x:f.temperature)std::printf("%a ",double(x));
  std::puts("");
 }
}
'''
    source = '#include <initializer_list>\n' + source
    before = build(tmp_path, 'original_adapter', original_root, source)
    after = build(tmp_path, 'corrected_adapter', ROOT, source)
    assert run(after) == run(before)
