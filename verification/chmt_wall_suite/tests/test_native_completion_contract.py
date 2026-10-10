"""Generated-input and execution-evidence contracts, not GPU physics evidence."""
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / 'verification/chmt_wall_suite'
sys.path.insert(0, str(SUITE))


def load(name):
    spec = importlib.util.spec_from_file_location('native_contract_' + name, SUITE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wall_couette_uses_supported_second_order_numerics_and_original_gate(tmp_path):
    case = tmp_path / 'wall'
    spec = load('generate').prepare('wall_constant_transport', case)
    schemes = (case / 'system/fvSchemes').read_text()
    assert 'fluxScheme Kurganov;' in schemes
    for key in ('phi,U', 'phid,p', 'phi,e', 'phi,K', 'phi,(p|rho)'):
        assert 'div(' + key + ') Gauss MUSCL;' in schemes
    assert 'limitedLinear' not in schemes and 'SLAU2' not in schemes
    assert spec['cells'] == 128
    assert spec['limits'] == {'velocity_l2': .02, 'velocity_linf': .05}
    assert spec['models']['gas'] == 'frozen equal-thermo'


def test_canonical_wall_gate_cannot_be_replaced_by_refinement(tmp_path):
    with pytest.raises(ValueError, match='128'):
        load('generate').prepare('wall_constant_transport', tmp_path / 'refined', {'cells': 256})


@pytest.mark.parametrize('name', ['gas_species_wave', 'gas_reacting_wave', 'flatplate_fixed', 'mss7_fixed'])
def test_shared_templates_declare_compatible_flux_and_reconstruction(tmp_path, name):
    case = tmp_path / name
    load('generate').prepare(name, case, {'nx': 4, 'ny': 4})
    schemes = (case / 'system/fvSchemes').read_text()
    assert 'fluxScheme Kurganov;' in schemes
    assert 'HLLC' not in schemes
    assert 'div(phi,U) Gauss MUSCL;' in schemes


@pytest.mark.parametrize('name', ['small_couette', 'small_sod', 'wall_constant_transport',
                                  'gas_species_wave', 'gas_reacting_wave', 'gas_chemistry_reactor',
                                  'flatplate_fixed', 'mss7_fixed'])
def test_gas_generators_schedule_exact_final_output(tmp_path, name):
    spec = load('generate').prepare(name, tmp_path / name, {'nx': 4, 'ny': 4})
    assert spec['time_controls']['writeControl'] == 'adjustableRunTime'
    assert int(spec['time_controls']['timePrecision']) >= 15
    assert set(('rho', 'p', 'T', 'U')).issubset(spec['completion_fields'])
    assert float(spec['time_controls']['endTime']) == spec['end_time']
    assert 0 < float(spec['time_controls']['writeInterval']) <= spec['end_time']


def final_fields(case, spec, *, time=None):
    directory = case / format(spec['end_time'] if time is None else time, '.17g')
    directory.mkdir(parents=True, exist_ok=True)
    for name in spec['completion_fields']:
        value = '(0 0 0)' if name == 'U' else '300' if name == 'T' else '1'
        (directory / name).write_text('internalField uniform ' + value + ';\n')
    return directory


@pytest.mark.parametrize('name', ['small_couette', 'small_sod'])
def test_legacy_completion_requires_exit_progress_and_exact_fields_without_end(tmp_path, name):
    case = tmp_path / name
    spec = load('generate').prepare(name, case)
    final_fields(case, spec)
    (case / 'log.solver').write_text(f"runTime = 1 simulationTime = {spec['end_time']:.17g} particleCount = 0 CoMax = .3\n")
    result = load('run').validate_completion(case, spec, solver_returncode=0)
    assert result['end_marker'] == 'legacy_final_progress'
    assert result['field_time'] == spec['end_time']
    assert result['checked_fields'] == spec['completion_fields']


@pytest.mark.parametrize('fault', ['missing_exit', 'failed_exit', 'missing_progress', 'early_progress',
                                  'nonfinite_progress', 'missing_field', 'nonfinite_field',
                                  'wrong_field_shape', 'nearby_time_only', 'ambiguous_time'])
def test_legacy_completion_rejects_incomplete_evidence(tmp_path, fault):
    case = tmp_path / 'case'
    spec = load('generate').prepare('small_couette', case)
    directory = final_fields(case, spec, time=.049997758186 if fault == 'nearby_time_only' else None)
    log = 'runTime = 1 simulationTime = .05 particleCount = 0\n'
    if fault == 'missing_progress': log = 'End\n'
    if fault == 'early_progress': log = 'runTime = 1 simulationTime = .0499 particleCount = 0\nEnd\n'
    if fault == 'nonfinite_progress': log = 'runTime = 1 simulationTime = nan particleCount = 0\n'
    if fault == 'missing_field': (directory / 'rho').unlink()
    if fault == 'nonfinite_field': (directory / 'T').write_text('internalField uniform nan;\n')
    if fault == 'wrong_field_shape': (directory / 'U').write_text('internalField nonuniform List<vector> 1 ((0 0 0));\n')
    if fault == 'ambiguous_time': (case / '0.05').mkdir()
    (case / 'log.solver').write_text(log)
    code = None if fault == 'missing_exit' else 1 if fault == 'failed_exit' else 0
    with pytest.raises((ValueError, OSError)):
        load('run').validate_completion(case, spec, solver_returncode=code)


def test_shared_completion_still_requires_end_and_backend_identity(tmp_path):
    case = tmp_path / 'wave'
    spec = load('generate').prepare('gas_reacting_wave', case, {'cells': 8})
    final_fields(case, spec)
    log = f"Shared gas model: api=1 Ns=2 mode=2\nrunTime = 1 simulationTime = {spec['end_time']:.17g} particleCount = 0\n"
    runner = load('run')
    (case / 'log.solver').write_text(log)
    with pytest.raises(ValueError, match='complete'):
        runner.validate_completion(case, spec, solver_returncode=0)
    (case / 'log.solver').write_text(log.replace('mode=2', 'mode=1') + 'End\n')
    with pytest.raises(ValueError, match='identity'):
        runner.validate_completion(case, spec, solver_returncode=0)
    (case / 'log.solver').write_text(log + 'End\n')
    assert runner.validate_completion(case, spec, solver_returncode=0)['end_marker'] == 'End'


@pytest.mark.parametrize('selection', ['default', 'absolute', 'relative', 'bare_relative'])
def test_backend_identity_matches_production_execl_resolution(tmp_path, selection):
    runner = load('run'); case = tmp_path / 'case'; case.mkdir()
    install = tmp_path / 'install'; install.mkdir()
    solver = install / 'gasUGKP'; solver.write_bytes(b'frontend fixture')
    backend = install / 'gasUGKPCudaBackend'; env = {}
    if selection == 'absolute': env['GAS_UGKP_CUDA_BACKEND'] = str(tmp_path / 'selected')
    if selection == 'relative': env['GAS_UGKP_CUDA_BACKEND'] = '../selected'
    if selection == 'bare_relative': env['GAS_UGKP_CUDA_BACKEND'] = 'selected'
    if selection != 'default':
        value = Path(env['GAS_UGKP_CUDA_BACKEND']); backend = value if value.is_absolute() else case / value
    backend.write_bytes(b'backend ' + selection.encode())
    record = runner.gas_backend_identity(solver, case, env)
    assert record['path'] == str(backend.resolve())
    assert record['sha256'] == hashlib.sha256(backend.read_bytes()).hexdigest()
    assert record['status'] == 'AVAILABLE'


@pytest.mark.parametrize('mutate_backend', [False, True])
def test_runner_pins_and_checks_the_actually_selected_backend(tmp_path, monkeypatch, mutate_backend):
    case = tmp_path / 'case'; spec = load('generate').prepare('small_couette', case)
    solver = tmp_path / 'gasUGKP'; solver.write_bytes(b'frontend fixture')
    backend = tmp_path / 'selected_backend'; backend.write_bytes(b'backend before run')
    monkeypatch.setenv('GAS_UGKP_CUDA_BACKEND', '../selected_backend')
    runner = load('run'); calls = []
    monkeypatch.setattr(runner, 'cuda_probe', lambda: {'status': 'AVAILABLE'})
    monkeypatch.setattr(runner.shutil, 'which', lambda name: '/fixture/' + name)
    monkeypatch.setattr(runner, 'compare', lambda path: {'acceptance': 'PASS'})

    def command(args, cwd, stdout, stderr, **kwargs):
        if args[0] == str(solver):
            calls.append(kwargs['env']['GAS_UGKP_CUDA_BACKEND'])
            final_fields(case, spec)
            stdout.write('runTime = 1 simulationTime = .05 particleCount = 0\n')
            if mutate_backend: backend.write_bytes(b'backend changed during run')
        else: stdout.write('Mesh OK.\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', command)
    result = runner.execute(case, 'execute', str(solver), allow_solver_fallback=False)
    assert calls == [str(backend.resolve())]
    assert result['backend']['path'] == str(backend.resolve())
    assert result['outcome'] == ('FAIL' if mutate_backend else 'PASS')
    if mutate_backend: assert 'changed' in result['reason']
    else: assert result['backend']['sha256_after'] == result['backend']['sha256']


@pytest.fixture(scope='module')
def real_frontend(tmp_path_factory):
    if not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('source OpenFOAM for the actual dictionary/capability reader')
    path = ROOT / 'applications/gasUGKP/tests/test_native_boundary_layer_frontend.py'
    spec = importlib.util.spec_from_file_location('native_frontend_dependency', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.native_wall_probe.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize('name,options', [('wall_constant_transport', {}),
                                         ('gas_species_wave', {'cells': 8}),
                                         ('gas_reacting_wave', {'cells': 8})] +
                                        [(name, {'wall': family, 'nx': 4, 'ny': 4})
                                         for name in ('flatplate_fixed', 'mss7_fixed')
                                         for family in ('lowRe', 'wallFunction', 'boundaryLayer_reactingSst')])
def test_generated_cases_pass_real_frontend_numerical_capabilities(tmp_path, real_frontend, name, options):
    case = tmp_path / name; spec = load('generate').prepare(name, case, options)
    for command in spec['mesh_commands']:
        mesh = subprocess.run([command[0], '-case', str(case)] + command[1:], capture_output=True, text=True)
        assert mesh.returncode == 0, mesh.stdout + mesh.stderr
    result = subprocess.run([str(real_frontend), '-case', str(case)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'WALL_READY' in result.stdout


@pytest.fixture(scope='module')
def real_time_probe(tmp_path_factory):
    if not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('source OpenFOAM for the actual Time scheduler')
    build = tmp_path_factory.mktemp('native_time_scheduler')
    (build / 'Make').mkdir()
    (build / 'probe.C').write_text(r'''#include "fvCFD.H"
using namespace Foam;
int main(int argc, char** argv) {
    #include "setRootCase.H"
    #include "createTime.H"
    const bool adjust=runTime.controlDict().lookupOrDefault<Switch>("adjustTimeStep",false);
    int steps=0;
    while(runTime.run()) {
        if(adjust) runTime.setDeltaT(min(runTime.deltaTValue()*1.137,scalar(.001731)));
        ++runTime;
        if(runTime.writeTime()) Info << "SCHEDULED_WRITE=" << runTime.timeName() << nl;
        if(++steps>100000) return 4;
    }
    Info << "FINAL_TIME=" << runTime.timeName() << nl;
}
''')
    (build / 'Make/files').write_text(f'probe.C\nEXE = {build}/probe\n')
    (build / 'Make/options').write_text('EXE_INC = -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    result = subprocess.run(['wmake'], cwd=build, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return build / 'probe'


@pytest.mark.parametrize('name', ['small_couette', 'small_sod', 'wall_constant_transport',
                                  'gas_species_wave', 'gas_chemistry_reactor', 'flatplate_fixed'])
def test_real_openfoam_scheduler_hits_requested_endpoint(tmp_path, real_time_probe, name):
    case = tmp_path / name
    spec = load('generate').prepare(name, case, {'nx': 4, 'ny': 4})
    result = subprocess.run([str(real_time_probe), '-case', str(case)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    writes = [float(value) for value in re.findall(r'SCHEDULED_WRITE=(\S+)', result.stdout)]
    endpoint = float(re.search(r'FINAL_TIME=(\S+)', result.stdout).group(1))
    tolerance = 1e-10 * max(abs(spec['end_time']), 1e-8)
    assert abs(endpoint - spec['end_time']) <= tolerance
    assert writes and abs(writes[-1] - spec['end_time']) <= tolerance
    if name == 'gas_chemistry_reactor':
        assert len(writes) == round(spec['end_time'] / float(spec['time_controls']['writeInterval']))
        assert all(any(abs(actual-target) <= tolerance for actual in [0.] + writes)
                   for target in spec['parameters']['check_times'])


def test_real_scheduler_exposes_old_runtime_write_gap(tmp_path, real_time_probe):
    case = tmp_path / 'legacy_schedule_counterexample'
    spec = load('generate').prepare('small_couette', case)
    load('generate').replace_entry(case / 'system/controlDict', 'writeControl', 'runTime')
    result = subprocess.run([str(real_time_probe), '-case', str(case)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    writes = [float(value) for value in re.findall(r'SCHEDULED_WRITE=(\S+)', result.stdout)]
    assert not any(abs(actual-spec['end_time']) <= 1e-10*spec['end_time'] for actual in writes)


def test_default_backend_follows_resolved_frontend_symlink_and_empty_override(tmp_path):
    install = tmp_path / 'install'; install.mkdir()
    solver = install / 'gasUGKP'; solver.write_bytes(b'frontend')
    backend = install / 'gasUGKPCudaBackend'; backend.write_bytes(b'backend')
    alias = tmp_path / 'gasUGKP-alias'; alias.symlink_to(solver)
    record = load('run').gas_backend_identity(alias, tmp_path, {'GAS_UGKP_CUDA_BACKEND': ''})
    assert record['path'] == str(backend)
    assert record['selection'] == 'frontend_sibling'


def test_missing_selected_backend_never_falls_back(tmp_path):
    solver = tmp_path / 'gasUGKP'; solver.write_bytes(b'frontend')
    (tmp_path / 'gasUGKPCudaBackend').write_bytes(b'unselected default')
    record = load('run').gas_backend_identity(solver, tmp_path, {'GAS_UGKP_CUDA_BACKEND': 'missing'})
    assert record['path'] == str(tmp_path / 'missing')
    assert record['status'] == 'UNAVAILABLE'
    assert 'sha256' not in record


@pytest.mark.parametrize('name', ['small_couette', 'small_sod', 'wall_constant_transport'])
def test_completion_contract_requires_created_conserved_energy(tmp_path, name):
    case = tmp_path / name
    spec = load('generate').prepare(name, case)
    # rhoE is READ_IF_PRESENT/AUTO_WRITE even when the initial template omits it.
    assert 'rhoE' in spec['completion_fields']
    directory = final_fields(case, spec)
    (directory / 'rhoE').unlink()
    identity = 'Shared gas model: api=1 Ns=2 mode=1\n' if name == 'wall_constant_transport' else ''
    (case / 'log.solver').write_text(identity + f'simulationTime = {spec["end_time"]:.17g}\nEnd\n')
    with pytest.raises(OSError):
        load('run').validate_completion(case, spec, solver_returncode=0)


def test_path_lookup_frontend_is_resolved_before_changing_to_case_directory(tmp_path, monkeypatch):
    case = tmp_path / 'case'; spec = load('generate').prepare('small_couette', case)
    install = tmp_path / 'install'; install.mkdir()
    solver = install / 'gasUGKP'; solver.write_bytes(b'frontend')
    backend = install / 'gasUGKPCudaBackend'; backend.write_bytes(b'backend')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GAS_UGKP_CUDA_BACKEND', raising=False)
    runner = load('run'); invocations = []
    monkeypatch.setattr(runner, 'cuda_probe', lambda: {'status': 'AVAILABLE'})
    monkeypatch.setattr(runner.shutil, 'which', lambda name: 'install/gasUGKP' if name == 'gasUGKP' else '/fixture/' + name)
    monkeypatch.setattr(runner, 'compare', lambda path: {'acceptance': 'PASS'})

    def command(args, cwd, stdout, stderr, **kwargs):
        if args[0].endswith('gasUGKP'):
            invocations.append(args[0])
            assert Path(args[0]).is_absolute()
            assert kwargs['env']['GAS_UGKP_CUDA_BACKEND'] == str(backend)
            final_fields(case, spec)
            stdout.write('simulationTime = .05\n')
        else: stdout.write('Mesh OK.\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', command)
    result = runner.execute(case, 'execute')
    assert result['outcome'] == 'PASS'
    assert invocations == [str(solver)]
    assert result['solver_path'] == str(solver)


@pytest.mark.parametrize('recorded', [False, True])
def test_pair_report_preserves_backend_provenance_separately_from_frontend(tmp_path, recorded):
    # Synthetic field/mesh/status fixtures exercise report metadata only.
    cases = []
    for index, family in enumerate(('lowRe', 'wallFunction')):
        case = tmp_path / family
        spec = load('generate').prepare('flatplate_fixed', case, {'nx': 4, 'ny': 4, 'wall': family})
        final_fields(case, spec)
        mesh = case / 'constant/polyMesh'; mesh.mkdir()
        for name in ('points', 'faces', 'owner', 'neighbour'): (mesh / name).write_text('identical fixture')
        status = {'native_execution': 'COMPLETED', 'solver_sha256': 'same frontend fixture'}
        if recorded: status['backend'] = {'status': 'AVAILABLE', 'path': f'/fixture/backend{index}',
                                          'sha256': str(index) * 64, 'sha256_after': str(index) * 64}
        (case / 'run_status.json').write_text(json.dumps(status))
        cases.append(case)
    report = load('compare_pair').compare_pair(*cases)
    for side, index in [('reference', 0), ('candidate', 1)]:
        record = report[side + '_backend']
        assert record['status'] == ('AVAILABLE' if recorded else 'UNAVAILABLE')
        if recorded:
            assert record['path'] == f'/fixture/backend{index}'
            assert record['sha256'] == str(index) * 64
        else: assert 'sha256' not in record
