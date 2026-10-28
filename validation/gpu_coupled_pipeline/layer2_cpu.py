#!/usr/bin/env python3
"""Isolated existing OpenFOAM dry-material CPU oracles for layer two.

Source OpenFOAM10 before running. The original driver and generators are copied
or imported unchanged from --source; no generated files enter that checkout.
This is an offline numerical oracle, not a GPU performance measurement.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(rows):
    def selected(mode):
        return [r for r in rows if r['mode'] == mode]
    def value(row, key):
        x = float(row[key])
        assert math.isfinite(x), (row, key)
        return x
    def orders(errors):
        return [math.log(a / b, 2) for a, b in zip(errors, errors[1:])]
    result = {}
    for mode in ('isothermal', 'adiabatic'):
        rr = selected(mode)
        order = orders([value(r, 'parent_error') for r in rr])
        assert all(.99 < x < 1.03 for x in order), (mode, order)
        assert max(abs(value(r, 'mass_residual')) / 500.5 for r in rr) < 1e-12
        assert max(abs(value(r, 'energy_residual')) / 350185000 for r in rr) < 1e-12
        assert max(abs(value(r, 'element_residual')) for r in rr) < 1e-10
        result[mode] = {'first_order_time_convergence': order}
    rr = selected('surface')
    for r in rr:
        assert value(r, 'wall_T_error') < 1e-7
        assert value(r, 'rate_error') < 1e-11
        assert value(r, 'recession_error') < 1e-14
        assert abs(value(r, 'energy_residual')) < 1e-6
        assert abs(value(r, 'element_residual')) < 1e-12
    rr = selected('darcy')
    order = orders([value(r, 'relative_flux_error') for r in rr])
    assert all(.95 < x < 1.05 for x in order)
    assert max(value(r, 'core_flux_difference') for r in rr) < 1e-14
    result['darcy'] = {'first_order_flux_convergence': order}
    rr = selected('mixture')
    order = orders([value(r, 'L2_cellaverage') for r in rr])
    assert all(x > 0 for x in order) and all(.95 < x < 1.05 for x in order[-2:])
    assert max(value(r, 'zero_sum') for r in rr) < 1e-14
    assert max(value(r, 'core_difference') for r in rr) < 1e-14
    result['unequal_diffusion'] = {'spatial_orders': order, 'asymptotic_grid': '80/160/320'}
    for r in selected('zeroSupport'):
        assert value(r, 'initial_new_flux') < 0 and value(r, 'old_centered_counterexample') > 0
        assert value(r, 'min_inventory') >= 0
        assert value(r, 'species_residual') < 1e-12
        assert value(r, 'energy_relative') < 1e-12
        assert value(r, 'temperature_error') < 1e-8
        assert value(r, 'zero_sum_flux') < 1e-12
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    source, work = args.source.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    original = source / 'validation/openfoam10_dry'
    shutil.copytree(original / 'driver', work / 'driver', ignore=shutil.ignore_patterns('linux*', '*.dep', '*.o'))
    (work / 'bin').mkdir()
    env = dict(os.environ, DRY_CORE_ROOT=str(source), DRY_VALIDATION_BIN=str(work / 'bin'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    assert env.get('WM_PROJECT_VERSION') == '10', 'Source OpenFOAM10 first'
    provenance = {
        'source_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_status': subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
        'source_root': str(source),
        'scope': 'Existing synthetic CPU oracles; no conduction migration and no native GPU speed claim.',
        'reference_sources': {str(p.relative_to(source)): digest(p) for p in sorted(original.rglob('*')) if p.is_file() and p.suffix in ('.C', '.H', '.py')},
        'command': ['wmake'],
    }
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with (work / 'build.log').open('w') as log:
        subprocess.run(['wmake'], cwd=work / 'driver', env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    provenance['binary_sha256'] = digest(work / 'bin/dryMaterialFoam')
    (work / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    spec = importlib.util.spec_from_file_location('unchanged_dry_cases', original / 'run_cases.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = work
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    rows = []
    matrix = [(mode, 8, dt, 1) for mode in ('isothermal', 'adiabatic') for dt in (.02, .01, .005)]
    matrix += [('surface', n, .01, 1) for n in (10, 20, 40)]
    matrix += [('darcy', n, .01, 1) for n in (20, 40, 80)]
    matrix += [('mixture', n, .01, 1) for n in (20, 40, 80, 160, 320)]
    matrix += [('zeroSupport', n, .0001, .01) for n in (8, 16)]
    for mode, n, dt, end in matrix:
        row = module.case(mode, n, dt, end)
        rows.append(row)
        (work / 'partial-results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    report = {'status': 'PASS', 'provenance_sha256': digest(work / 'provenance.json'), 'cases': rows, 'assessment': assess(rows)}
    (work / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('PASS: 19 existing dry CPU oracle cases', flush=True)


if __name__ == '__main__':
    main()