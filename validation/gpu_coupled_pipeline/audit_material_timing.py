#!/usr/bin/env python3
"""Audit durable native material timings without rerunning any solver."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import layer1
import layer2_native
from repeat_case import input_hash


def audit_reacting_wall(case, material):
    result = layer1.audit_chmt(case, closed=False)
    result['scope'] = 'Open gas boundaries; packet channels and sealed material mass/energy balance. Global gas inventory changes are observations, not a closed-domain assertion.'
    previous = None; residuals = {'mass': 0., 'energy': 0.}
    for name, rows in material.items():
        step = int(name.split('-')[-1])
        current = {'mass': math.fsum(v for row in rows for key, v in row.items() if key.startswith(('condensed_', 'pore_'))),
                   'energy': math.fsum(row['energy'] for row in rows)}
        wall = layer1.physical_rows(case / 'chmtOutput' / f'exchange-{step}.csv')
        if previous:
            for key in residuals:
                difference = abs(current[key] - previous[key] + math.fsum(row[key] for row in wall))
                residuals[key] = max(residuals[key], difference)
                tolerance = max(2e-5, 2e-13 * abs(previous[key])) if key == 'energy' else 1e-10 * max(1., abs(previous[key]))
                assert difference <= tolerance, (case, key, difference)
        previous = current
    result['material_exchange_residuals'] = residuals
    return result


def one(cohort, name, inspector, scenario):
    case = cohort / name; reference = cohort / 'baseline-0'
    raw = (case / 'timing.txt').read_text().splitlines()
    assert len(raw) == 2 and int(raw[1]) == 0, (name, raw)
    elapsed = float(raw[0]); assert math.isfinite(elapsed) and elapsed > 0
    assert input_hash(case) == input_hash(reference), (name, 'input mismatch')
    current = layer2_native.checkpoint_material(case, inspector)
    expected = layer2_native.checkpoint_material(reference, inspector)
    budget = audit_reacting_wall(case, current) if scenario == 'reacting-wall' else layer2_native.audit(case, scenario, current)
    last = max(int(k.split('-')[-1]) for k in current)
    final_time = layer1.physical_rows(case / 'chmtOutput' / f'accepted-{last}.csv')[0]['time']
    analytic_error = None
    if scenario == 'dry':
        analytic_error = abs(budget['states'][-1]['condensed_0'] - 500 * math.exp(-2 * final_time))
        assert analytic_error < .002, analytic_error
    result = dict(status='PASS', elapsed_seconds=elapsed, input_sha256=input_hash(case),
                  parity=layer1.compare_chmt(reference, case),
                  material_maximum_scaled_errors=layer2_native.compare_material(expected, current),
                  audit=budget, pyrolysis_analytic_error=analytic_error)
    (case / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--inspector', type=Path, required=True)
    p.add_argument('--scenario', choices=('dry','receding','reacting-wall'), default='dry')
    p.add_argument('--exploratory', action='store_true', help='Report one un-warmed pair separately from formal repetitions')
    p.add_argument('--run', help='Audit one already completed run in a fresh process')
    a = p.parse_args(); a.cohort = a.cohort.resolve(); a.inspector = a.inspector.resolve()
    if a.run:
        one(a.cohort, a.run, a.inspector, a.scenario); return
    assert (a.cohort / 'solver-runs-complete').read_text().strip() == 'complete'
    with (a.cohort / 'order.tsv').open() as stream: order = list(csv.DictReader(stream, delimiter='\t'))
    report = dict(status='AUDITING', timing_scope=(a.cohort / 'timing-method.txt').read_text().strip(),
                  binaries=(a.cohort / 'binaries.sha256').read_text().splitlines(), runs=[], performance={})
    def save(): (a.cohort / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    for row in order:
        case = a.cohort / row['case']; path = case / 'audit.json'
        if not path.exists():
            # Audits are isolated from one another and from the original timers.
            # Preserve failed audit logs; rerunning this command never reruns a solver.
            attempt = len(list(case.glob('audit-attempt-*.log'))) + 1
            with (case / f'audit-attempt-{attempt}.log').open('w') as log:
                completed = subprocess.run([sys.executable, __file__, '--cohort', str(a.cohort),
                    '--inspector', str(a.inspector), '--scenario', a.scenario, '--run', row['case']], stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                report.update(status='AUDIT_FAILED', failed_run=row['case'], audit_exit_code=completed.returncode)
                save(); raise SystemExit(completed.returncode)
        item = json.loads(path.read_text()); assert item['status'] == 'PASS'
        item.update(kind=row['kind'], repeat=int(row['repeat']), warmup=int(row['repeat']) == 0)
        report['runs'].append(item); save()
    samples = {k: [r['elapsed_seconds'] for r in report['runs'] if r['kind'] == k and not r['warmup']] for k in ('baseline', 'candidate')}
    if a.exploratory:
        assert len(report['runs']) == 2 and all(r['warmup'] for r in report['runs'])
        elapsed = {r['kind']: r['elapsed_seconds'] for r in report['runs']}
        report['performance'] = dict(scope='Exploratory single pair, no warmup, not a repeated-timing estimate',
            elapsed_seconds=elapsed, speedup=elapsed['baseline']/elapsed['candidate'])
        report['status'] = 'PASS'; save(); print(json.dumps(report['performance']), flush=True); return
    assert len(samples['baseline']) == len(samples['candidate']) and len(samples['baseline']) >= 3
    medians = {k: statistics.median(v) for k, v in samples.items()}
    report['performance'] = dict(samples_seconds=samples, median_seconds=medians,
        speedup=medians['baseline']/medians['candidate'])
    report['status'] = 'PASS'; save(); print(json.dumps(report['performance']), flush=True)


if __name__ == '__main__':
    main()
