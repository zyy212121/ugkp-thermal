"""Offline field probe generation follows the actual production constructors.

These Python cases never invoke the C++/CUDA compiler or execute a solver.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = SOURCE_ROOT
SPEC = importlib.util.spec_from_file_location('particle_contract_tool', (SOURCE_ROOT / 'tool/particle_field_contract.py'))
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture(scope='session')
def source_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('particle-contract')
    paths = ['common/ParticleFieldManifest.json', 'common/GpuParticleFields.cuh',
             'common/GpuParticleRestartFields.inl', 'common/GpuParticleSavedVelocityRestart.inl',
             'common/GpuParticleBufferCommit.cuh']
    paths += [f'applications/{app}/{leaf}/GpuResidentStrict.cu' for app, leaf in
              [('gasUGKP', 'private_backend'), ('FSH', 'private_backend'), ('CHT', 'gpu')]]
    for relative in paths:
        data = (ROOT / relative).read_bytes()
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


@pytest.mark.parametrize('app', ['gasUGKP', 'FSH', 'CHT'])
def test_actual_constructor_generates_complete_probe(source_root, app):
    meta = json.loads((source_root / 'common/ParticleFieldManifest.json').read_text())
    code = TOOL.build_probe(source_root, meta, app)
    assert 'PASS typed lifecycle/copy/swap/transfer' in code
    if app == 'gasUGKP':
        return
    source = (source_root / f'applications/{app}/' /
              ('gpu' if app == 'CHT' else 'private_backend') / 'GpuResidentStrict.cu').read_text()
    constructor = TOOL.function_body(source, 'ugkwpGpuResidentStrictConfigureParticleStuckModel')
    cold_aliases = re.findall(r'^\s*([\w ]+)\*&\s*(\w+)\s*=\s*s->((?:compactP|p)Cold\w+)\s*;',
                             constructor, re.M)
    assert len(cold_aliases) == 14
    for scalar, alias, field in cold_aliases:
        assert f'{scalar.strip()}*& {alias}=s->{field};' in code
        allocation = [call for call in TOOL.calls(constructor, 'acquire')
                      if re.match(r'acquire\s*\(\s*' + alias + r'\s*,', call)]
        assert len(allocation) == 1
        assert allocation[0] + ';' in code
        initialisation = [call for call in TOOL.calls(constructor, 'initialise')
                          if re.match(r'initialise\s*\(\s*' + alias + r'\s*,', call)]
        for call in initialisation:
            assert call + ';' in code
    for helper in ('pending', 'markPending', 'clearPending', 'acquire', 'initialise'):
        assert re.search(r'auto\s+' + helper + r'\s*=', code)
    assert 'if (owner != nullptr) return 0;' in code
    assert 'if (!pending(bit)) return cudaSuccess;' in code


def test_generated_registry_and_wire_order_remain_exact(source_root):
    meta = json.loads((source_root / 'common/ParticleFieldManifest.json').read_text())
    assert TOOL.registry(meta) == (source_root / 'common/GpuParticleFields.cuh').read_text()
    assert TOOL.restart_fields(meta) == (source_root / 'common/GpuParticleRestartFields.inl').read_text()


def test_helper_probe_keeps_actual_initialisation_byte_checks(source_root):
    meta = json.loads((source_root / 'common/ParticleFieldManifest.json').read_text())
    code = TOOL.build_probe(source_root, meta, 'CHT')
    assert 'coldNodeStorage*sizeof(float)' in code
    assert 'cold2DNodeStorage*sizeof(float)' in code
    assert 'particleCapacity*sizeof(*coldContactAge)' in code
    assert 'memset byte closure' in code
    assert 'initialization changed retained cold storage' in code


@pytest.mark.parametrize('failure', ['missing', 'duplicate'])
def test_missing_or_duplicate_actual_acquisition_is_still_reported(source_root, tmp_path, failure):
    root = tmp_path / 'source'
    for path in source_root.rglob('*'):
        if path.is_file():
            destination = root / path.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    path = root / 'applications/FSH/private_backend/GpuResidentStrict.cu'
    source = path.read_text()
    call = next(call for call in TOOL.calls(source, 'acquire')
                if re.match(r'acquire\s*\(\s*coldFrozenArea\s*,', call))
    source = source.replace(call, '0' if failure == 'missing' else call + ' != 0 || ' + call, 1)
    path.write_text(source)
    meta = json.loads((root / 'common/ParticleFieldManifest.json').read_text())
    with pytest.raises(ValueError, match=r'FSH: allocation closure for pColdFrozenArea: expected one, got [02]'):
        TOOL.build_probe(root, meta, 'FSH')


def migration_tool():
    spec = importlib.util.spec_from_file_location('scheduling_migration_tool', (SOURCE_ROOT / 'tool/migrate_scheduling_properties.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('existing_schedule', [False, True])
@pytest.mark.parametrize('legacy', [
    'gpuResearchVariant experimental;\ngpuCsrLevel L1;\n',
    'gpuCsrLevel unknown;\ngpuCsrHeavyReduction true;\n',
    'gpuCsrLevel auto;\ngpuCsrHeavyReductionAutoInterval -7;\n',
])
def test_migration_preserves_current_successful_scheduling_inputs(tmp_path, existing_schedule, legacy):
    module = migration_tool()
    particle = tmp_path / 'particleProperties'
    particle.write_text('rhoSolid 3000;\n' + ('' if existing_schedule else legacy))
    schedule = particle.with_name('schedulingProperties')
    if existing_schedule:
        schedule.write_text(legacy)
    module.migrate(particle)
    result = schedule.read_text()
    expected = 'L1' if 'gpuResearchVariant' in legacy else 'L0' if 'unknown' in legacy else 'auto'
    assert module.scalar_value(result, 'gpuCsrLevel') == expected
    if expected == 'auto':
        assert module.scalar_value(result, 'gpuCsrHeavyReductionAutoInterval') == '-7'
    assert 'gpuResearchVariant' not in particle.read_text() + result
    assert particle.read_text() == 'rhoSolid 3000;\n'


def test_migration_keeps_actual_zero_divisor_failure_without_partial_writes(tmp_path):
    module = migration_tool()
    particle = tmp_path / 'particleProperties'
    original = 'rhoSolid 3000;\ngpuCsrLevel auto;\ngpuCsrHeavyReductionAutoInterval 0;\n'
    particle.write_text(original)
    with pytest.raises(ValueError, match='zero modulo divisor'):
        module.migrate(particle)
    assert particle.read_text() == original
    assert not particle.with_name('schedulingProperties').exists()


@pytest.mark.parametrize('legacy,expected', [
    ('gpuCsrHeavyReduction true;\n', 'L2'),
    ('gpuCsrHeavyReduction auto;\n', 'auto'),
    ('gpuCsrCellLocalPath true;\n', 'L1'),
    ('', 'L0'),
])
def test_migration_retains_legacy_schedule_translation(tmp_path, legacy, expected):
    module = migration_tool()
    particle = tmp_path / 'particleProperties'
    particle.write_text('rhoSolid 3000;\n' + legacy)
    module.migrate(particle)
    assert module.scalar_value(particle.with_name('schedulingProperties').read_text(), 'gpuCsrLevel') == expected
