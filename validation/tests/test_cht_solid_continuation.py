"""Execute the actual OpenFOAM thermal owners; no CUDA device is emulated here."""
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
            source=(THERMAL/(name+'.C')).read_text().replace('linearPerformance = equation.solve();','linearPerformance = equation.solve(); t14LinearResidual=linearPerformance.finalResidual(); t14LinearIterations=linearPerformance.nIterations(); t14LinearConverged=linearPerformance.converged();')
            (build/f'{name}.C').write_text('double t14LinearResidual=0; int t14LinearIterations=0; bool t14LinearConverged=false;\n'+source)
        elif name=='GpuThermalExchangeState':(build/f'{name}.C').write_text(observe_restart((THERMAL/(name+'.C')).read_text()))
        else:(build/f'{name}.C').write_text(f'#include "{THERMAL/name}.C"\n')
    (build/'Make/files').write_text('probe.C\n'+'\n'.join(n+'.C' for n in owners)+f'\nEXE = {build}/probe\n')
    (build/'Make/options').write_text(f'EXE_INC = -I{THERMAL} -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
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

@pytest.mark.parametrize('mode',['property_overflow','diagnostic_overflow','ordinary','atomic_directory','linear_one','nearest_rounding','properties','unordered','reversed_axis','duplicate_endpoint','clamp','nonoverlap','signed','absorbed','candidate_one','candidate_zero','rebind','mapper','pending','state','state_machine'])
def test_actual_thermal_continuation(thermal_probe,tmp_path,mode):
    p=execute(thermal_probe,tmp_path,mode)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'PASS '+mode in p.stdout

@pytest.mark.parametrize('mode',['missing_property','missing_state','missing_pending','truncated_pending','missing_conservative','atomic_io'])
def test_actual_required_data_and_io_errors(thermal_probe,tmp_path,mode):
    p=execute(thermal_probe,tmp_path,mode)
    assert p.returncode==0,p.stdout+p.stderr
    assert 'PASS '+mode in p.stdout
