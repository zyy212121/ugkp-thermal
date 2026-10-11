"""Regression contracts for generated inputs and species-specific execution."""
import importlib.util
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / 'verification/chmt_wall_suite'
sys.path.insert(0, str(SUITE))


def load(name):
    spec = importlib.util.spec_from_file_location('review_' + name, SUITE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_replace_inline_entry_without_changing_stop_at_value(tmp_path):
    path = tmp_path / 'controlDict'
    path.write_text('startFrom startTime; startTime 0; stopAt endTime; endTime 0.01; deltaT 1e-5;\n')
    load('foam').replace_entry(path, 'endTime', .025)
    assert path.read_text() == 'startFrom startTime; startTime 0; stopAt endTime; endTime 0.025; deltaT 1e-5;\n'


def test_replace_entry_rejects_duplicate_keys_without_writing(tmp_path):
    path = tmp_path / 'controlDict'
    original = 'endTime 0.01; endTime 0.025;\n'
    path.write_text(original)
    with pytest.raises(ValueError, match='duplicate'):
        load('foam').replace_entry(path, 'endTime', .05)
    assert path.read_text() == original


def test_replace_entry_preserves_comments_and_nested_dictionary(tmp_path):
    path = tmp_path / 'controlDict'
    original = '// endTime 99;\nFoamFile { object "endTime"; }\nother {\nendTime 88;\n}\n/* endTime 77; */ endTime 0.01;\n'
    path.write_text(original)
    load('foam').replace_entry(path, 'endTime', .025)
    assert path.read_text() == original.replace('endTime 0.01;', 'endTime 0.025;')


def test_generated_chmt_override_has_single_correct_end_time(tmp_path):
    case = tmp_path / 'slab'
    spec = load('generate').prepare('chmt_receding_slab', case, {'end_time': .025})
    text = (case / 'system/controlDict').read_text()
    assert re.findall(r'\bendTime\s+([^;]+);', text) == ['0.025']
    assert float(spec['time_controls']['endTime']) == spec['end_time'] == .025


def test_actual_openfoam_reads_requested_chmt_end_time(tmp_path):
    command = shutil.which('foamDictionary')
    if command is None:
        pytest.skip('source the OpenFOAM environment to run the real dictionary reader')
    case = tmp_path / 'slab'
    load('generate').prepare('chmt_receding_slab', case, {'end_time': .025})
    result = subprocess.run([command, str(case / 'system/controlDict'), '-entry', 'endTime', '-value'],
                            check=True, capture_output=True, text=True)
    assert float(result.stdout.strip()) == .025


def test_suite_missing_ns10_never_looks_up_ns2_binary(tmp_path, monkeypatch):
    import run as runner
    suite = load('suite')
    case = tmp_path / 'reactor'
    load('generate').prepare('gas_chemistry_reactor', case, {})
    gas2 = tmp_path / 'gasUGKP'
    gas2.write_text('not invoked')
    item = dict(id='gas_chemistry_reactor', path=str(case), category=2, gate='report_only',
                solver='gasUGKP', species_count=10, native_execution='NOT_RUN')
    monkeypatch.setattr(suite, 'prepare_suite', lambda *a: {'cases': [item], 'unsupported': []})
    monkeypatch.setattr(runner, 'cuda_probe', lambda: {'status': 'AVAILABLE'})
    lookups = []
    commands = []

    def which(name):
        lookups.append(name)
        return str(gas2) if name == 'gasUGKP' else '/available/' + name

    def execute(args, *, cwd, stdout, stderr):
        commands.append(args)
        stdout.write('Mesh OK.\n' if args[0] == 'checkMesh' else 'Shared gas model: api=1 Ns=2 mode=1\nEnd\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.shutil, 'which', which)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr(sys, 'argv', ['suite.py', '--output', str(tmp_path), '--mode', 'execute',
                                    '--cases', 'gas_chemistry_reactor', '--gas-solver2', str(gas2)])
    with pytest.raises(SystemExit) as result:
        suite.main()
    assert result.value.code == 3
    assert 'gasUGKP' not in lookups
    assert not any(args[0] == str(gas2) for args in commands)
    status = json.loads((case / 'run_status.json').read_text())
    assert status['native_execution'] == status['outcome'] == 'NOT_RUN'


def test_contact_card_has_no_invented_element_or_pyrolysis_mapping(tmp_path):
    spec = load('generate').prepare('small_cht_contact', tmp_path / 'contact', {})
    card = spec['material_card']
    assert card['surface_reactions'] == card['gas_phase_reactions'] == []
    assert [s['molar_mass'] for s in card['gas_species']] == [.028, .032]
    assert 'not declared' in card['element_basis'].lower()
    assert 'disabled' in card['pyrolysis_products'].lower()


def test_imported_mesh_count_includes_neighbour_only_cell(tmp_path):
    mesh = tmp_path / 'mesh'
    shutil.copytree(ROOT / 'examples/thermal/MSS7_laminar/constant/fluid/polyMesh', mesh)
    # Focused addressing fixture: the highest label occurs only in neighbour.
    (mesh / 'owner').write_text('FoamFile { format ascii; class labelList; object owner; }\n3\n(0 1 2)\n')
    (mesh / 'neighbour').write_text('FoamFile { format ascii; class labelList; object neighbour; }\n2\n(1 3)\n')
    spec = load('generate').prepare('mss7_fixed', tmp_path / 'case',
                                    {'gas_mesh': str(mesh), 'nx': 2, 'ny': 3})
    assert spec['cells'] == 4


def _exact_couette_output(case):
    """Deliberate oracle fixture for evidence-guard tests, never native evidence."""
    metrics = load('metrics')
    spec = load('generate').prepare('wall_constant_transport', case, {})
    count = spec['cells']
    directory = case / str(spec['end_time'])
    directory.mkdir()
    velocity = [metrics.couette_average(i / count, (i + 1) / count, .05, 1, 1, .1)
                for i in range(count)]
    (directory / 'U').write_text('internalField nonuniform List<vector>\n' + str(count) + '\n(\n' +
                                '\n'.join(f'({value:.17g} 0 0)' for value in velocity) + '\n);\n')
    return spec


def _wall_csv(case, time=.05, fault=None):
    keys = ('face owner matchingDistance ownerDistance heatFluxIntoGas tractionX tractionY tractionZ '
            'wallKFlux integratedKSource ownerOmega residual iterations traceDensity viscosity stageTime '
            'speciesFlux_A speciesFlux_B reactionIntegral_A reactionIntegral_B').split()
    rows = []
    for face in (0, 1):
        row = dict.fromkeys(keys, 0.)
        row.update(face=face, owner=face, matchingDistance=.02, ownerDistance=.01,
                   traceDensity=1., viscosity=.1, stageTime=.049)
        rows.append(row)
    if fault == 'missing_column':
        keys.remove('stageTime')
        for row in rows:
            row.pop('stageTime')
    elif fault == 'nonfinite':
        rows[0]['heatFluxIntoGas'] = float('nan')
    elif fault == 'future_stage':
        rows[0]['stageTime'] = .06
    elif fault == 'initial_stage_only':
        for row in rows:
            row['stageTime'] = 0.
    elif fault == 'missing_face':
        rows.pop()
    elif fault == 'invalid_distance':
        rows[0]['ownerDistance'] = .03
    path = case / str(time) / 'uniform/gasBoundaryLayer.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_couette_without_wall_execution_evidence_cannot_pass(tmp_path):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    with pytest.raises(ValueError, match='wall'):
        load('metrics').compare(case)


@pytest.mark.parametrize('fault', ['missing_column', 'nonfinite', 'future_stage',
                                  'initial_stage_only', 'missing_face', 'invalid_distance', 'stale_write'])
def test_new_wall_hard_case_rejects_invalid_native_diagnostic(tmp_path, fault):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    _wall_csv(case, time=.04 if fault == 'stale_write' else .05, fault=fault)
    with pytest.raises(ValueError):
        load('metrics').compare(case)


def test_valid_new_wall_evidence_preserves_hard_gate_and_lowre_unavailable(tmp_path):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    _wall_csv(case)
    assert load('metrics').compare(case)['acceptance'] == 'PASS'
    low = load('metrics').wall_diagnostics(tmp_path / 'low', {'id': 'flatplate_fixed'})
    assert low['status'] == low['yplus'] == 'UNAVAILABLE'


BUDGET_CHANNELS = ('budgetTransportK', 'budgetTransportOmega', 'budgetSourceK',
                   'budgetSourceOmega', 'budgetConstraintK', 'budgetConstraintOmega')
BUDGET_FIELDS = ('budgetIntervalStart', 'budgetIntervalDuration', *BUDGET_CHANNELS)


def _budget_extension(case, availability):
    _wall_csv(case)
    path = case / '0.05/uniform/gasBoundaryLayer.csv'
    with path.open() as stream:
        reader = csv.DictReader(stream)
        keys, rows = list(reader.fieldnames), list(reader)
    keys += ['budgetAuditAvailable', *BUDGET_FIELDS]
    for index, (row, flag) in enumerate(zip(rows, availability)):
        row['budgetAuditAvailable'] = flag
        row.update({key: str(index + 1) if flag else '' for key in BUDGET_FIELDS})
    return path, keys, rows


def _write_budget_extension(path, keys, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize('availability', [(0, 0), (1, 1), (0, 1)])
def test_optional_budget_channels_preserve_required_wall_evidence(tmp_path, availability):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    path, keys, rows = _budget_extension(case, availability)
    _write_budget_extension(path, keys, rows)
    metrics = load('metrics')
    assert metrics.compare(case)['acceptance'] == 'PASS'
    diagnostic = metrics.wall_diagnostics(case, {})
    budget = diagnostic['budget_audit']
    assert budget['available_rows'] == sum(availability)
    assert budget['unavailable_rows'] == len(availability) - sum(availability)
    assert budget['status'] == ('AVAILABLE' if any(availability) else 'UNAVAILABLE')
    if any(availability):
        values = [i + 1 for i, flag in enumerate(availability) if flag]
        assert budget['statistics']['budgetSourceK']['mean_available_owner_rows'] == sum(values) / len(values)
        assert budget['units']['budgetSourceK'] == 'J'
        assert budget['units']['budgetSourceOmega'] == 'kg/s'
        assert sum(interval['available_owner_rows'] for interval in budget['intervals']) == sum(availability)
    else:
        assert budget['statistics'] == {}
    assert not any(key in diagnostic['statistics'] for key in BUDGET_FIELDS)


@pytest.mark.parametrize('fault', ['invalid_flag', 'blank_flag', 'nonfinite_flag',
                                  'unavailable_nonblank', 'unavailable_nan',
                                  'available_blank', 'available_nan',
                                  'missing_channel', 'missing_flag', 'missing_interval',
                                  'nonfinite_interval', 'negative_duration', 'duplicate_owner'])
def test_optional_budget_rejects_invalid_or_hidden_evidence(tmp_path, fault):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    path, keys, rows = _budget_extension(case, (0, 1))
    if fault == 'invalid_flag':
        rows[0]['budgetAuditAvailable'] = '0.5'
    elif fault == 'blank_flag':
        rows[0]['budgetAuditAvailable'] = ''
    elif fault == 'nonfinite_flag':
        rows[0]['budgetAuditAvailable'] = 'nan'
    elif fault.startswith('unavailable_'):
        rows[0]['budgetSourceK'] = 'nan' if fault.endswith('nan') else '0'
    elif fault.startswith('available_'):
        rows[1]['budgetSourceK'] = 'nan' if fault.endswith('nan') else ''
    elif fault == 'nonfinite_interval':
        rows[1]['budgetIntervalDuration'] = 'nan'
    elif fault == 'negative_duration':
        rows[1]['budgetIntervalDuration'] = '-1'
    elif fault == 'duplicate_owner':
        rows[0].update({key: '1' for key in BUDGET_FIELDS})
        rows[0]['budgetAuditAvailable'] = 1
        rows[1]['owner'] = rows[0]['owner']
    else:
        key = {'missing_channel': 'budgetSourceK', 'missing_flag': 'budgetAuditAvailable',
               'missing_interval': 'budgetIntervalStart'}[fault]
        keys.remove(key)
        for row in rows:
            row.pop(key)
    _write_budget_extension(path, keys, rows)
    with pytest.raises(ValueError, match='budget'):
        load('metrics').compare(case)


def test_optional_budget_interval_does_not_guess_rk_stage_relationship(tmp_path):
    case = tmp_path / 'case'
    _exact_couette_output(case)
    path, keys, rows = _budget_extension(case, (1, 1))
    for row in rows:
        row.update(budgetIntervalStart='.0495', budgetIntervalDuration='0')
    _write_budget_extension(path, keys, rows)
    budget = load('metrics').wall_diagnostics(case, {})['budget_audit']
    assert budget['intervals'] == [{'start': .0495, 'duration': 0., 'available_owner_rows': 2}]
