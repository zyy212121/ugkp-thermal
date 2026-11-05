"""Execute the actual OpenFOAM thermal owners; no CUDA device is emulated here."""
from validation.tools.cht_thermal_cuda import link_thermal_cuda
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import os
import subprocess
import shutil
import re
import pytest

ROOT=SOURCE_ROOT
THERMAL=Path(os.environ.get('T14_THERMAL_SOURCE',ROOT/'applications/CHT/thermal'))
SPEC=importlib.util.spec_from_file_location('t14_surface_cases',(VALIDATION_ROOT / 'tests/test_cht_surface_temperature.py'))
CASES=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CASES)

def observe_restart(source):
    """Count actual .C open/helper executions; never alter production sources."""
    source=source.replace('Foam::IFstream stream(path);','Foam::IFstream stream(path); ++t14DictionaryOpens;')
    for symbol in ['canonicalConfigurationSha1','canonicalMeshTopologySha1','canonicalWallTemperatureSha1','fileSha1','validateFiniteRestartFields']:
        counter='t14FieldScans' if symbol=='validateFiniteRestartFields' else 't14HashCalls'
        source=re.sub(r'('+symbol+r'\s*\([^{}]*\)\s*\{)',lambda m:m[1]+' ++'+counter+';',source,count=1)
    return 'int t14DictionaryOpens=0, t14HashCalls=0, t14FieldScans=0;\n'+source

@pytest.fixture(scope='module')
def thermal_probe(tmp_path_factory):
    if not (shutil.which('wmake') and shutil.which('blockMesh')):
        pytest.skip('Foundation OpenFOAM required')
    build=tmp_path_factory.mktemp('thermal_native');(build/'Make').mkdir()
    shutil.copy((VALIDATION_ROOT / 'tests/fixtures/cht_thermal/probe.C'),build/'probe.C')
    owners=['GpuSolidThermalProperties','GpuSolidThermalCoupler','GpuSolidPatchMapper','GpuThermalExchangeState']
    for name in owners:
        if name=='GpuSolidThermalCoupler':
            source=(THERMAL/(name+'.C')).read_text().replace('linearPerformance = ugkp::gpu::solveFoamScalarMatrix(equation, linearControls, *sparse_);','linearPerformance = ugkp::gpu::solveFoamScalarMatrix(equation, linearControls, *sparse_); if (std::getenv("UGKP_TEST_LINEAR_NAN")) linearPerformance.finalResidual() = std::numeric_limits<scalar>::quiet_NaN(); t14LinearResidual=linearPerformance.finalResidual(); t14LinearIterations=linearPerformance.nIterations(); t14LinearConverged=linearPerformance.converged();')
            (build/f'{name}.C').write_text('double t14LinearResidual=0; int t14LinearIterations=0; bool t14LinearConverged=false;\n'+source)
        elif name=='GpuThermalExchangeState':(build/f'{name}.C').write_text(observe_restart((THERMAL/(name+'.C')).read_text()))
        else:(build/f'{name}.C').write_text(f'#include "{THERMAL/name}.C"\n')
    (build/'Make/files').write_text('probe.C\n'+'\n'.join(n+'.C' for n in owners)+f'\nEXE = {build}/probe\n')
    (build/'Make/options').write_text(f'EXE_INC = -I{THERMAL} -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    link_thermal_cuda(build, THERMAL.parents[2])
    p=subprocess.run(['wmake'],cwd=build,text=True,capture_output=True)
    (build/'build.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stdout+p.stderr
    return build/'probe'

def execute(probe,path,mode,extra=''):
    CASES.make_case(path,1000,variable=False,transverse_cells=4 if mode=='nearest_rounding' else 3 if mode in ('linear_one','diagnostic_overflow') else 1)
    if mode=='diagnostic_overflow':
        f=path/'0/T';f.write_text(f.read_text().replace('right { type zeroGradient;', 'right { type fixedValue; value uniform 300;'))
    if mode=='nearest_rounding':
        coarse=path/'coarse_seed';CASES.make_case(coarse,0,transverse_cells=2);(path/'constant/coarse').mkdir();shutil.copytree(coarse/'constant/polyMesh',path/'constant/coarse/polyMesh');(path/'system/coarse').mkdir()
        for f in ['fvSchemes','fvSolution']:shutil.copy(coarse/'system'/f,path/'system/coarse'/f)
    if mode=='linear_one':
        f=path/'system/blockMeshDict';f.write_text(f.read_text().replace('(1 3 1)','(3 3 1)'));p=subprocess.run(['blockMesh','-case',str(path)],capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
        f=path/'system/fvSolution';f.write_text(f.read_text().replace('tolerance 1e-14;','tolerance 0; maxIter 1;'))
    with (path/'constant/testProperties').open('a') as f:f.write(f't14Mode {mode};\n'+extra)
    env=os.environ.copy();env.pop('FOAM_SIGFPE',None)
    return subprocess.run([str(probe),'-case',str(path)],text=True,capture_output=True,env=env)

@pytest.mark.parametrize('mode',['diagnostic_overflow','ordinary','atomic_directory','linear_one','nearest_rounding','properties','unordered','reversed_axis','duplicate_endpoint','clamp','nonoverlap','absorbed','candidate_one','candidate_zero','rebind','mapper','pending','state','state_machine'])
def test_actual_thermal_continuation(thermal_probe,tmp_path,mode):
    p=execute(thermal_probe,tmp_path,mode)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'PASS '+mode in p.stdout

@pytest.mark.parametrize('mode',['missing_property','missing_state','missing_pending','truncated_pending','missing_conservative','atomic_io'])
def test_actual_required_data_and_io_errors(thermal_probe,tmp_path,mode):
    p=execute(thermal_probe,tmp_path,mode)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'PASS '+mode in p.stdout


@pytest.mark.parametrize('mode',['property_overflow','signed','negative_temperature','nan_temperature','zero_property'])
def test_consumed_physical_errors(thermal_probe,tmp_path,mode):
    p=execute(thermal_probe,tmp_path,mode)
    assert p.returncode != 0, p.stdout+p.stderr
    assert 'invalid consumed solid' in p.stdout+p.stderr

@pytest.mark.parametrize('mode',['ordinary','candidate_one','linear_one'])
def test_warning_output_only(thermal_probe,tmp_path,mode,monkeypatch):
    monkeypatch.delenv('UGKP_DIAGNOSTIC_WARNINGS',raising=False)
    enabled=execute(thermal_probe,tmp_path/'enabled',mode)
    monkeypatch.setenv('UGKP_DIAGNOSTIC_WARNINGS','0')
    disabled=execute(thermal_probe,tmp_path/'disabled',mode)
    assert enabled.returncode == disabled.returncode == 0, enabled.stdout+enabled.stderr+disabled.stdout+disabled.stderr
    if mode != 'ordinary':
        assert 'retaining computed candidate' in enabled.stdout+enabled.stderr
    assert 'retaining computed candidate' not in disabled.stdout+disabled.stderr
    def parity(output): return [line for line in output.splitlines() if line.startswith('PARITY')]
    assert parity(enabled.stdout) == parity(disabled.stdout)


def test_native_nonfinite_linear_error(thermal_probe,tmp_path,monkeypatch):
    monkeypatch.setenv('UGKP_TEST_LINEAR_NAN','1')
    p=execute(thermal_probe,tmp_path,'candidate_one')
    assert p.returncode != 0
    assert 'solid linear solve returned a nonfinite residual' in p.stdout+p.stderr
    assert 'finalResidual=nan' in p.stdout+p.stderr
    assert 'time=' in p.stdout+p.stderr
