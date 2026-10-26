"""Changed C05 paths only: true original/current FP64 thermal owner comparisons.

The resident/device boundary is explicit recording storage. These are not GPU tests.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import pytest

ROOT = SOURCE_ROOT
ORIGINAL = 'd6110c02745a8ad726208a426c4bb3a28233c1fd'
BASE = '038cbdac762fafc03ce60e9f94ae506ee155332d'
NAMES = ['GpuSolidThermalCoupler', 'GpuSolidThermalProperties', 'GpuSolidPatchMapper',
         'GpuThermalExchangeState', 'GpuParticleRadiationCoupler', 'MieDoRadiationSolver', 'MieTable']
spec = importlib.util.spec_from_file_location('c05_case_writer', (VALIDATION_ROOT / 'tests/test_cht_full_exchange.py'))
cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(cases)
EVIDENCE = Path(os.environ.get('C05_EVIDENCE', '/tmp/chmt-c05-evidence'))
BUILD_ROOT = Path(os.environ.get('C05_BUILD_ROOT', '/tmp/chmt-c05-builds'))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_changed(path, text):
    if not path.exists() or path.read_text() != text: path.write_text(text)

def build(revision):
    if not shutil.which('wmake'): pytest.skip('Foundation OpenFOAM required')
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    p = BUILD_ROOT/revision; p.mkdir(parents=True, exist_ok=True)
    tree = ROOT
    if revision != 'working':
        tree = p/'source'
        if not tree.exists():
            tree.mkdir()
            archive = subprocess.check_output(['git', 'archive', revision, 'applications/CHT/thermal', 'applications/CHT/gpu', 'common'], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as t: t.extractall(tree, filter="data")
    thermal = tree/'applications/CHT/thermal'
    for n in ['compatibility.C', 'CompatibilityResident.H', 'InterfaceExchangeResident.inc']:
        write_changed(p/n, ((VALIDATION_ROOT / 'tests/fixtures/cht_thermal')/n).read_text())
    for n in NAMES:
        text = (thermal/(n+'.C')).read_text()
        if n == 'GpuSolidThermalCoupler':
            text = text.replace('#include "../gpu/GpuResidentStrict.H"', '#include "CompatibilityResident.H"')
            needle = 'nonlinearIterations(iterations)\n{'
            assert text.count(needle) == 1
            text = '#include <iostream>\n'+text.replace(needle, needle+'\n    std::cout << "DATA candidateResidual " << std::hexfloat << residual << "\\nDATA candidateIterations " << iterations << "\\n";')
        write_changed(p/(n+'.C'), text)
    (p/'Make').mkdir(exist_ok=True)
    write_changed(p/'Make/files', 'compatibility.C\n'+'\n'.join(n+'.C' for n in NAMES)+f'\nEXE = {p}/probe\n')
    write_changed(p/'Make/options', f'EXE_INC = -std=c++17 -DUGKWP_USE_CUDA -I{thermal} -I{tree}/applications/CHT/gpu -I{tree}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    dependencies = {}
    for subtree in ['applications/CHT/thermal','applications/CHT/gpu','common']:
        for f in sorted((tree/subtree).rglob('*')):
            if f.is_file(): dependencies[str(f.relative_to(tree))] = sha(f)
    for n in ['compatibility.C','CompatibilityResident.H','InterfaceExchangeResident.inc','Make/files','Make/options']:
        dependencies['fixture/'+n] = sha(p/n)
    signature = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
    receipt_path = p/'receipt.json'
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    reused = old.get('signature') == signature and (p/'probe').exists() and old.get('binary_sha256') == sha(p/'probe')
    elapsed = 0
    if not reused:
        start = time.monotonic(); result = subprocess.run(['wmake'], cwd=p, capture_output=True, text=True); elapsed = time.monotonic()-start
        (EVIDENCE/f'build-{revision}.log').write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stdout+result.stderr
    receipt = dict(revision=revision, signature=signature, dependencies=dependencies, binary=str(p/'probe'), binary_sha256=sha(p/'probe'), reused=reused, compile_wall_s=elapsed, flags='WM_DP 64=64', resident_boundary='recording; GpuReal double; no CUDA execution')
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    (EVIDENCE/f'build-{revision}.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return p/'probe'

@pytest.fixture(scope='module')
def probes(): return {r: build(r) for r in [ORIGINAL, os.environ.get('C05_REVISION','working')]}

def make(path, mode, options=''):
    seed = 'outer_one' if mode in ['routing_extra','routing_two'] else 'radiation' if 'clock' in mode or mode=='flags_radiation' else 'pending_reordered' if mode.startswith('decode_') else 'mismatch'
    cases.make_case(path, seed)
    config = path/'constant/exchangeConfig'; text = config.read_text()
    if mode.startswith('routing_'):
        text = text.replace('solidInterfaceMaxIterations 1;', 'solidInterfaceMaxIterations 200;').replace('solidInterfaceTemperatureRelativeTolerance 0;', 'solidInterfaceTemperatureRelativeTolerance 1e-10;').replace('solidInterfaceEnergyRelativeTolerance 0;', 'solidInterfaceEnergyRelativeTolerance 1e-10;')
    if mode == 'routing_extra':
        props = 'rho {type constant; value 1000;} Cp {type constant; value 1;} kappa {type constant; value 6;} emissivity {type constant; value 0.8;}'
        text = text.replace('formatVersion 2;', 'formatVersion 1; solidRegion solid; mapping oneToOneConformal; coupledInterfaces ({fluidPatch left;solidPatch right;}); properties {'+props+'}')
    if mode.startswith('legacy'):
        options += f' reviewStateVersion {1 if mode.startswith("legacy1") else 2}; reviewExtraRadiationClock -0.1;'
    if mode.startswith('surface_'):
        t = path/'0/T'; t.write_text(t.read_text().replace('left { type fixedValue; value uniform 300;', 'left { type fixedGradient; gradient uniform 0;'))
    config.write_text(text+'\nc05Mode '+mode+';\n'+options+'\n')

def run(binary, path, mode, options=''):
    make(path, mode, options)
    result = subprocess.run([str(binary), '-case', str(path)], capture_output=True, text=True)
    text = result.stdout+result.stderr
    label = binary.parent.name
    (EVIDENCE/(mode+'-'+label+'.log')).write_text(text)
    records = [s for s in result.stdout.splitlines() if s.startswith('DATA ')]
    return result, records

@pytest.mark.parametrize('mode', ['routing_extra','routing_two','legacy1_extra_clock','legacy2_extra_clock','flags_disabled','flags_off','flags_radiation','surface_ordinary'])
def test_original_success_exact(probes,tmp_path,mode):
    revision = os.environ.get('C05_REVISION','working')
    old, a = run(probes[ORIGINAL], tmp_path/'original', mode)
    new, b = run(probes[revision], tmp_path/'current', mode)
    assert old.returncode == 0, old.stdout+old.stderr
    assert new.returncode == 0, new.stdout+new.stderr
    assert a and a == b, '\n'.join(['ORIGINAL']+a+['CURRENT']+b)

@pytest.mark.parametrize('mode', ['surface_owner','surface_gradient','surface_delta'])
def test_saved_surface_ignores_unused_certificate(probes,tmp_path,mode):
    revision = os.environ.get('C05_REVISION','working')
    old,_ = run(probes[ORIGINAL],tmp_path/'original',mode)
    new,b = run(probes[revision],tmp_path/'current',mode)
    assert old.returncode != 0
    assert new.returncode == 0,new.stdout+new.stderr
    assert b == ['DATA savedSurface.0 0x1.2cp+8']

DECODERS = [
 ('untagged_present','',True,None),
 ('text_present','stateVersion futureTag;',True,None),
 ('zero_present','stateVersion "0";',True,None),
 ('negative_present','stateVersion "-1";',True,None),
 ('unknown_present','stateVersion "99";',True,None),
 ('untagged_absent','pending false;',False,'pending wall-energy ledger'),
 ('text_absent','stateVersion futureTag; pending false;',False,'pending wall-energy ledger'),
 ('unknown_absent','stateVersion "99"; pending false;',False,'pending wall-energy ledger'),
 ('current_absent','stateVersion "4"; pending false;',False,'pending wall-energy ledger'),
 ('zero_missing_clock','stateVersion "0"; radiationClock false;',False,'completedRadiationSimulationTimeS'),
 ('negative_missing_clock','stateVersion "-1"; radiationClock false;',False,'completedRadiationSimulationTimeS'),
 ('text_missing_clock','stateVersion futureTag; radiationClock false;',False,'completedRadiationSimulationTimeS'),
 ('current_missing_clock','stateVersion "4"; radiationClock false;',False,'completedRadiationSimulationTimeS'),
 ('legacy1_present','stateVersion "1";',True,None),
 ('legacy2_present','stateVersion "2";',True,None),
 ('legacy1_absent','stateVersion "1"; radiationClock false; pending false;',True,None),
 ('legacy2_absent','stateVersion "2"; radiationClock false; pending false;',True,None),
 ('legacy3_absent','stateVersion "3"; pending false;',True,None),
 ('missing_slot','stateVersion futureTag; missingSlot true;',False,'missing a required configured face'),
 ('missing_rhoE','stateVersion futureTag; missingRhoE true;',False,'rhoE'),
]
@pytest.mark.parametrize('name,options,success,error', DECODERS)
def test_actual_decoder_inventory(probes,tmp_path,name,options,success,error):
    revision = os.environ.get('C05_REVISION','working')
    result, records = run(probes[revision],tmp_path,name if name.startswith('decode_') else 'decode_'+name,options)
    if not success:
        assert result.returncode != 0
        assert error in result.stderr,result.stdout+result.stderr
    else:
        assert result.returncode == 0,result.stdout+result.stderr
        d = dict(line.split()[1:] for line in records)
        shared = name.startswith(('legacy1','legacy2'))
        assert float.fromhex(d['decodedClock']) == (-.2 if shared else -.3)
        absent = 'pending false;' in options
        assert float.fromhex(d['restoreCalls']) == (0 if absent else 1)
        for i in range(3):
            for key, value in [('gas',50),('deposited',20),('reflected',-5)]:
                assert float.fromhex(d[f'{key}.{i}']) == (0 if absent else value)

@pytest.mark.parametrize('three,tag', [(False, 'futureTag'), (True, 'futureTag'), (False, None), (True, None)])
def test_unknown_identity_uses_actual_layout(probes,tmp_path,three,tag):
    revision = os.environ.get('C05_REVISION','working')
    reference_mode = 'routing_two' if three else 'flags_off'
    old, a = run(probes[ORIGINAL],tmp_path/'original',reference_mode)
    path = tmp_path/'current'; make(path,reference_mode)
    config = path/'constant/exchangeConfig'; text = config.read_text()
    original_tag = 'formatVersion 2;' if three else 'formatVersion 1;'
    text = text.replace(original_tag, '' if tag is None else f'formatVersion {tag};',1)
    config.write_text(text)
    result = subprocess.run([str(probes[revision]),'-case',str(path)],capture_output=True,text=True)
    label = f'layout-{three}-{tag}'
    (EVIDENCE/(label+'-'+revision+'.log')).write_text(result.stdout+result.stderr)
    b = [line for line in result.stdout.splitlines() if line.startswith('DATA ')]
    assert old.returncode == result.returncode == 0, old.stderr+result.stderr
    assert a == b
