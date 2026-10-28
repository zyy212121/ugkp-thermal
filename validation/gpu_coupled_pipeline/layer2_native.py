#!/usr/bin/env python3
"""Layer-two native dry-material parity, physical budgets and elapsed time.

Uses the original openfoam10_dry runner/generators and its analytic checks. Only
its external schema-8 checker command is redirected to the schema-9 adapter;
all numerical assertions remain unchanged. Every solver run is fresh. Failed
runs remain in results.json and abort the cohort without silent retry.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import layer1
from repeat_case import input_hash


def checkpoint_material(case, inspector):
    result = {}
    for path in sorted((case / 'chmtOutput').glob('checkpoint-*/state.chmt'), key=lambda p: int(p.parent.name.split('-')[-1])):
        text = subprocess.check_output([str(inspector), str(path.parent)], text=True)
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(text.splitlines())]
        assert all(math.isfinite(v) for r in rows for v in r.values())
        result[path.parent.name] = rows
    return result


def compare_material(left, right):
    assert left.keys() == right.keys()
    maximum = {}
    for step in left:
        assert len(left[step]) == len(right[step])
        for a, b in zip(left[step], right[step]):
            assert a.keys() == b.keys()
            for key, x in a.items():
                y = b[key]
                if key == 'cell':
                    assert x == y
                error = abs(x - y)
                scale = max(1., abs(x), abs(y))
                maximum[key] = max(maximum.get(key, 0.), error / scale)
                assert error <= 2e-10 * scale, (step, key, x, y)
    return maximum


def audit(case, scenario, material):
    # The current selected fixtures have a closed outer boundary. Conversion
    # changes individual species; the synthetic dry stoichiometric invariants
    # below account for those changes explicitly.
    states = []
    maximum = {}
    for name, solid in material.items():
        step = int(name.split('-')[-1])
        out = case / 'chmtOutput'
        gas = layer1.physical_rows(out / f'gas-{step}.csv')
        wall = layer1.physical_rows(out / f'exchange-{step}.csv')
        total = lambda rows, key: math.fsum(r[key] for r in rows)
        state = {
            'step': step,
            'mass': total(gas, 'mass') + math.fsum(v for r in solid for k, v in r.items() if k.startswith(('condensed_', 'pore_'))),
            'gas_mass': total(gas, 'mass'),
            'gas_energy': total(gas, 'total_energy'),
            'material_energy': total(solid, 'energy'),
            'condensed_0': total(solid, 'condensed_0'),
            'condensed_1': total(solid, 'condensed_1'),
            'progress_0': total(solid, 'progress_0'),
            'pore_mass': math.fsum(v for r in solid for k, v in r.items() if k.startswith('pore_')),
            'volume': total(solid, 'volume'),
        }
        state['energy'] = state['gas_energy'] + state['material_energy']
        assert all(math.isfinite(v) for v in state.values())
        assert all(v >= -1e-12 for r in solid for k, v in r.items() if k.startswith(('condensed_', 'pore_', 'progress_')))
        if scenario == 'dry':
            state['reaction_progress_invariant'] = state['condensed_0'] + state['progress_0']
            state['C1_yield_invariant'] = state['condensed_1'] + .3 * state['condensed_0']
            state['S0_yield_invariant'] = total(gas, 'mass_S0') + total(solid, 'pore_0') + .2 * state['condensed_0']
            state['S1_yield_invariant'] = total(gas, 'mass_S1') + total(solid, 'pore_1') + .5 * state['condensed_0']
        if states:
            previous, initial = states[-1], states[0]
            for key in ('mass', 'energy') + tuple(k for k in state if k.endswith('_invariant')):
                residual = abs(state[key] - initial[key])
                maximum[key] = max(maximum.get(key, 0.), residual)
                tolerance = max(2e-5, 2e-13 * abs(initial[key])) if key == 'energy' else 1e-10 * max(1., abs(initial[key]))
                assert residual <= tolerance, (case, key, residual)
            for key, state_key, sign in [('mass', 'gas_mass', 1), ('energy', 'gas_energy', 1), ('energy', 'material_energy', -1)]:
                residual = abs(state[state_key] - previous[state_key] - sign * total(wall, key))
                maximum[state_key + '_exchange'] = max(maximum.get(state_key + '_exchange', 0.), residual)
                tolerance = 2e-5 if key == 'energy' else 1e-10 * max(1., initial['mass'])
                assert residual <= tolerance, (case, state_key, residual)
        states.append(state)
    return {'maximum_absolute_residuals': maximum, 'states': states, 'scope': 'Closed inventories, signed interface mass/energy, and dry-case stoichiometric yields/progress; source-rate CPU/GPU parity is covered separately by kernel oracles.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-role',default='Accepted layer-one GPU-interface executable, with material still on CPU.')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--source', type=Path, required=True, help='Frozen source containing original validation runner')
    parser.add_argument('--cases-root', type=Path, required=True)
    parser.add_argument('--inspector', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--scenarios', nargs='+', default=['thermal', 'single', 'diffusion', 'dry'])
    parser.add_argument('--grid', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=0)
    parser.add_argument('--end-time', type=float)
    args = parser.parse_args()
    args.work = args.work.resolve(); args.source = args.source.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    paths = {k: getattr(args, k).resolve() for k in ('baseline', 'candidate') if getattr(args, k)}
    os.environ.update(CHMT_CUDA_STACK_BYTES='65536', OMP_NUM_THREADS='4', UGKP_CASES_ROOT=str(args.cases_root.resolve()))
    original = args.source / 'validation/openfoam10_dry/native/run_native.py'
    spec = importlib.util.spec_from_file_location('frozen_native_runner', original)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    original_run = runner.run
    def run_with_schema9(command, log, env):
        if len(command) > 1 and Path(command[1]).name == 'check_coupled.py':
            command = [command[0], Path(__file__).with_name('check_coupled_schema9.py'), *command[2:]]
        return original_run(command, log, env)
    runner.run = run_with_schema9
    report = {
        'status': 'running', 'grid': args.grid,
        'baseline_role': args.baseline_role,
        'binaries': {k: {'path': str(v), 'sha256': layer1.digest(v)} for k, v in paths.items()},
        'runner_source': {'path': str(original), 'sha256': layer1.digest(original)},
        'inspector_sha256': layer1.digest(args.inspector),
        'timing_scope': 'Complete solver process including startup and normal output; generation, mesh and audits excluded. Alternating order, one warmup pair excluded.',
        'runs': [], 'performance': {},
    }
    def save():
        (args.work / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    for scenario in args.scenarios:
        reference = None; reference_material = None; reference_input = None
        samples = {k: [] for k in paths}
        for repeat in range(args.repeats + 1):
            order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
            for kind in order:
                target = args.work / f'{scenario}-{kind}-{repeat}'
                record = {'scenario': scenario, 'kind': kind, 'repeat': repeat, 'warmup': repeat == 0, 'directory': target.name}
                report['runs'].append(record); save()
                argv = [str(original), '--binary', str(paths[kind]), '--source-root', str(args.source), '--cases-root', str(args.cases_root.resolve()), '--output', str(target), '--scenarios', scenario, '--grid', str(args.grid)]
                if args.end_time is not None:
                    argv += ['--end-time', str(args.end_time)]
                previous_argv = sys.argv
                try:
                    sys.argv = argv
                    runner.main()
                    raw = json.loads((target / 'results.json').read_text())['results'][0]
                    case = target / f'{scenario}-0'
                    material = checkpoint_material(case, args.inspector)
                    snapshot_hash = input_hash(case)
                    if reference is None:
                        reference, reference_material, reference_input = case, material, snapshot_hash
                    assert snapshot_hash == reference_input, 'Input identity changed within a cohort'
                    record.update(status='PASS', elapsed_seconds=raw['elapsed_seconds'], input_sha256=snapshot_hash,
                        original_checks=raw, parity=layer1.compare_chmt(reference, case),
                        checkpoint_material_maximum_scaled_errors=compare_material(reference_material, material), audit=audit(case, scenario, material),
                        warnings={'foam_warning_count': (case / 'run.log').read_text().count('FOAM Warning')})
                    if repeat:
                        samples[kind].append(raw['elapsed_seconds'])
                except BaseException as error:
                    record.update(status='FAILED', error=repr(error)); report['status'] = 'FAILED'; save(); raise
                finally:
                    sys.argv = previous_argv
                save()
        medians = {k: statistics.median(v) for k, v in samples.items() if v}
        report['performance'][scenario] = {'samples_seconds': samples, 'median_seconds': medians}
        if 'baseline' in medians and 'candidate' in medians:
            report['performance'][scenario]['speedup'] = medians['baseline'] / medians['candidate']
    report['status'] = 'PASS'; save()
    print(json.dumps({'status': report['status'], 'runs': len(report['runs']), 'report': str(args.work / 'results.json')}), flush=True)


if __name__ == '__main__':
    main()
