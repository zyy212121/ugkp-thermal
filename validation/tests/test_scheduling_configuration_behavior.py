"""Exercise the real OpenFOAM dictionary parser and production scheduling header.

Checks formal bundles, automatic heavy scheduling and legacy input fallback.
No CUDA device or case computation is used.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os,subprocess
import pytest

ROOT=SOURCE_ROOT

@pytest.fixture(scope='module')
def parser(tmp_path_factory):
    out=tmp_path_factory.mktemp('thermal-scheduling')
    source=out/'parse.C'
    source.write_text(r"""
#include "GpuSchedulingConfiguration.H"
#include "gpu/GpuBoundarySchedule.H"
#include "IStringStream.H"
int main(int argc,char** argv) {
    Foam::IStringStream input(argv[1]);
    Foam::dictionary properties(input);
    if(argc>2){
      Foam::List<Foam::Tuple2<Foam::scalar,Foam::scalar>> rows(properties.lookup("rows"));
      Foam::GpuBoundaryScheduleTable table(rows);Foam::scalarField times,values;table.copyColumns(times,values);
      Foam::Info << "ROWS " << table.size();forAll(times,i)Foam::Info << ' ' << times[i] << ':' << values[i];Foam::Info << '\n';return 0;
    }
    const auto v=Foam::readGpuSchedulingConfiguration(properties);
    Foam::Info << "FLAGS " << int(v.csrLevel) << ' '
      << v.csrCellLocalPath << ' ' << v.warpAggregatedBinning << ' '
      << v.splitPreDirectory << ' ' << int(v.heavyReductionMode) << ' ' << v.heavyReductionAutoInterval << '\n';
}
""")
    foam=Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
    lib=Path(os.environ.get('FOAM_LIBBIN',str(foam/'platforms/linux64GccDPInt32Opt/lib')))
    exe=out/'parse'
    cmd=['g++','-std=c++17','-DWM_DP','-DWM_LABEL_SIZE=32','-DNoRepository','-I'+str(ROOT/'common'),
         '-I'+str(foam/'src/OpenFOAM/lnInclude'),'-I'+str(foam/'src/finiteVolume/lnInclude'),
         '-I'+str(foam/'src/meshTools/lnInclude'),'-I'+str(foam/'src/OSspecific/POSIX/lnInclude'),
         str(source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lfiniteVolume','-lmeshTools','-lOpenFOAM','-L'+str(lib/'dummy'),'-Wl,-rpath,'+str(lib/'dummy'),'-lPstream','-o',str(exe)]
    q=subprocess.run(cmd,capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    return exe

def run(parser,config):
    env=dict(os.environ);env.pop('FOAM_ABORT',None)
    return subprocess.run([str(parser),config],env=env,capture_output=True,text=True)

@pytest.mark.parametrize('level,flags',[('L0','0 0 0 0 0'),('L1','1 1 1 1 0'),('L2','2 1 1 1 1')])
def test_formal_levels_preserve_operator_bundles(parser,level,flags):
    q=run(parser,f'gpuCsrLevel {level};')
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS '+flags in q.stdout,q.stdout

@pytest.mark.parametrize('variant',['L0','L1','L2','E1','E2','T1','T2','S1','S2'])
def test_legacy_research_entry_does_not_override_selected_level(parser,variant):
    q=run(parser,f'gpuCsrLevel L2; gpuResearchVariant {variant};')
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS 2 1 1 1 1' in q.stdout

@pytest.mark.parametrize('level',['E1','E2','T1','T2','S1','S2','bad'])
def test_unknown_scheduling_labels_use_existing_l0(parser,level):
    q=run(parser,f'gpuCsrLevel {level};')
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS 0 0 0 0 0' in q.stdout

def test_unused_automatic_interval_does_not_reject_l1(parser):
    q=run(parser,'gpuCsrLevel L1; gpuCsrHeavyReductionAutoInterval 100;')
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS 1 1 1 1 0' in q.stdout

@pytest.mark.parametrize('config,expected', [('gpuCsrLevel E1;', 'L0'), ('gpuCsrLevel L2;\ngpuResearchVariant S2;', 'L2')])
def test_migration_keeps_successful_current_input_fallback(tmp_path,config,expected):
    import importlib.util
    spec=importlib.util.spec_from_file_location('thermal_migration_success',(SOURCE_ROOT / 'tool/migrate_scheduling_properties.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    particle=tmp_path/'particleProperties';particle.write_text('parcelMass 5e-9;\n')
    schedule=tmp_path/'schedulingProperties';schedule.write_text(config+'\n')
    module.migrate(particle)
    assert module.scalar_value(schedule.read_text(),'gpuCsrLevel')==expected
    assert 'gpuResearchVariant' not in schedule.read_text()+particle.read_text()
    assert particle.read_text()=='parcelMass 5e-9;\n'

def test_migration_reports_actual_zero_divisor_before_writing(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('thermal_migration',(SOURCE_ROOT / 'tool/migrate_scheduling_properties.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    particle=tmp_path/'particleProperties';particle.write_text('parcelMass 5e-9;\n')
    schedule=tmp_path/'schedulingProperties';schedule.write_text('gpuCsrLevel auto;\ngpuCsrHeavyReductionAutoInterval 0;\n')
    before=(particle.read_bytes(),schedule.read_bytes())
    with pytest.raises(ValueError,match='zero modulo divisor'):module.migrate(particle)
    assert (particle.read_bytes(),schedule.read_bytes())==before

@pytest.mark.parametrize('interval',[1,100,1000])
def test_auto_retains_L1_operations_and_selects_existing_automatic_mode(parser,interval):
    q=run(parser,f'gpuCsrLevel auto; gpuCsrHeavyReductionAutoInterval {interval};')
    assert q.returncode==0,q.stdout+q.stderr
    assert f'FLAGS 3 1 1 1 2 {interval}' in q.stdout,q.stdout

def test_auto_uses_existing_default_interval(parser):
    q=run(parser,'gpuCsrLevel auto;')
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS 3 1 1 1 2 100' in q.stdout

@pytest.mark.parametrize('interval',[0])
def test_auto_reports_undefined_zero_divisor(parser,interval):
    q=run(parser,f'gpuCsrLevel auto; gpuCsrHeavyReductionAutoInterval {interval};')
    assert q.returncode!=0
    assert 'gpuCsrHeavyReductionAutoInterval' in q.stdout+q.stderr

@pytest.mark.parametrize('legacy',[False,True])
def test_migration_preserves_auto_and_inspection_interval(tmp_path,legacy):
    import importlib.util
    spec=importlib.util.spec_from_file_location('thermal_auto_migration',(SOURCE_ROOT / 'tool/migrate_scheduling_properties.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    particle=tmp_path/'particleProperties';particle.write_text('parcelMass 5e-9;\n')
    schedule=tmp_path/'schedulingProperties'
    schedule.write_text(('gpuCsrHeavyReduction auto;' if legacy else 'gpuCsrLevel auto;')+'\ngpuCsrHeavyReductionAutoInterval 1000;\n')
    module.migrate(particle)
    assert module.scalar_value(schedule.read_text(),'gpuCsrLevel')=='auto'
    assert module.scalar_value(schedule.read_text(),'gpuCsrHeavyReductionAutoInterval')=='1000'
    assert 'parcelMass 5e-9;' in particle.read_text()

@pytest.mark.parametrize('config', ['', 'gpuResidentMaxFaceWalkHops 0;', 'gpuResidentMaxFaceWalkHops -7;', 'gpuResidentCourantUpdateInterval -3;', 'gpuResidentMaxDeltaTGrowth -0.5;', 'gpuPrecision 16;', 'gpuParticleBlockThreads 24;', 'gpuReductionBlockThreads 96;'])
def test_policy_only_fields_do_not_refuse_existing_schedule(parser,config):
    q=run(parser,config)
    assert q.returncode==0,q.stdout+q.stderr
    assert 'FLAGS 0 0 0 0 0' in q.stdout

@pytest.mark.parametrize('interval',[-1,-100])
def test_negative_auto_interval_retains_unsigned_modulo_input(parser,interval):
    q=run(parser,f'gpuCsrLevel auto; gpuCsrHeavyReductionAutoInterval {interval};')
    assert q.returncode==0,q.stdout+q.stderr
    assert f'FLAGS 3 1 1 1 2 {interval}' in q.stdout

def test_negative_actual_particle_capacity_remains_error(parser):
    q=run(parser,'gpuResidentParticleCapacity -1;')
    assert q.returncode!=0

@pytest.mark.parametrize('rows,expected', [('()', 'ROWS 0'), ('((-2 -3) (0 1))','ROWS 2 -2:-3 0:1'), ('((2 3) (1 -5) (1 7))','ROWS 3 2:3 1:-5 1:7')])
def test_table_propagates_actual_empty_signed_unsorted_columns(parser,rows,expected):
    q=subprocess.run([str(parser),f'rows {rows};','table'],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    assert expected in q.stdout

@pytest.mark.parametrize('rows',['((0))','((0 1) (1))'])
def test_table_missing_required_pair_member_still_errors(parser,rows):
    q=subprocess.run([str(parser),f'rows {rows};','table'],capture_output=True,text=True)
    assert q.returncode!=0
