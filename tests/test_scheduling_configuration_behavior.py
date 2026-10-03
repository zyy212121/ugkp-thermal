"""Exercise the real OpenFOAM dictionary parser and production scheduling header.

Catches reopened research/auto inputs and changes to the three formal bundles.
No CUDA device or case computation is used.
"""
from pathlib import Path
import os,subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def parser(tmp_path_factory):
    out=tmp_path_factory.mktemp('thermal-scheduling')
    source=out/'parse.C'
    source.write_text(r"""
#include "GpuSchedulingConfiguration.H"
#include "IStringStream.H"
int main(int argc,char** argv) {
    Foam::IStringStream input(argv[1]);
    Foam::dictionary properties(input);
    const auto v=Foam::readGpuSchedulingConfiguration(properties);
    Foam::Info << "FLAGS " << int(v.csrLevel) << ' '
      << v.csrCellLocalPath << ' ' << v.warpAggregatedBinning << ' '
      << v.splitPreDirectory << ' ' << int(v.heavyReductionMode) << '\n';
}
""")
    foam=Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
    lib=Path(os.environ.get('FOAM_LIBBIN',str(foam/'platforms/linux64GccDPInt32Opt/lib')))
    exe=out/'parse'
    cmd=['g++','-std=c++14','-DWM_DP','-DWM_LABEL_SIZE=32','-DNoRepository','-I'+str(ROOT/'common'),
         '-I'+str(foam/'src/OpenFOAM/lnInclude'),'-I'+str(foam/'src/finiteVolume/lnInclude'),
         '-I'+str(foam/'src/meshTools/lnInclude'),'-I'+str(foam/'src/OSspecific/POSIX/lnInclude'),
         str(source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lfiniteVolume','-lmeshTools','-lOpenFOAM','-o',str(exe)]
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
def test_research_entry_is_rejected_even_for_matching_levels(parser,variant):
    q=run(parser,f'gpuCsrLevel L2; gpuResearchVariant {variant};')
    assert q.returncode!=0,'thermal accepted research override: '+q.stdout
    assert 'gpuResearchVariant' in q.stdout+q.stderr

@pytest.mark.parametrize('level',['auto','E1','E2','T1','T2','S1','S2','bad'])
def test_only_three_formal_level_names_are_accepted(parser,level):
    q=run(parser,f'gpuCsrLevel {level};')
    assert q.returncode!=0,'thermal accepted non-formal level: '+q.stdout
    assert 'gpuCsrLevel' in q.stdout+q.stderr

def test_automatic_interval_is_rejected(parser):
    q=run(parser,'gpuCsrLevel L1; gpuCsrHeavyReductionAutoInterval 100;')
    assert q.returncode!=0,q.stdout
    assert 'gpuCsrHeavyReductionAutoInterval' in q.stdout+q.stderr

def test_migration_rejects_unsupported_inputs_before_writing(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('thermal_migration',ROOT/'tools/migrate_scheduling_properties.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for config in ['gpuCsrLevel auto;','gpuCsrLevel L2;\ngpuResearchVariant S2;','gpuCsrHeavyReduction auto;']:
        case=tmp_path/'constant';case.mkdir(exist_ok=True)
        particle=case/'particleProperties';particle.write_text('parcelMass 5e-9;\n')
        schedule=case/'schedulingProperties';schedule.write_text(config+'\n')
        before=(particle.read_bytes(),schedule.read_bytes())
        with pytest.raises(ValueError):module.migrate(particle)
        assert (particle.read_bytes(),schedule.read_bytes())==before
