#!/usr/bin/env python3
"""Compile and run local CUDA budgets against the repository's actual operators."""
from pathlib import Path
import argparse
import json
import resource
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Separate directory for generated test binaries and logs')
    parser.add_argument('--app', action='append', choices=['gasUGKP', 'FSH', 'CHT'], help='Default: all applications present in this checkout')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for contract in ('managed_mirrors.py', 'particle_field_contract.py', 'operator_policy_contract_test.py'):
        subprocess.run([sys.executable, str(root / 'tools' / contract)], check=True)
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error('--output must be outside the source checkout')
    apps = args.app or [a for a in ['gasUGKP', 'FSH', 'CHT'] if (root / 'applications' / a).is_dir()]
    for app in apps:
        if not (root / 'applications' / app).is_dir():
            parser.error(f'{app} is not present in this checkout')
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    requested = max(soft, 64 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_STACK, (requested if hard == resource.RLIM_INFINITY else min(requested, hard), hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output.mkdir(parents=True, exist_ok=True)
    fixture = root / 'tests' / 'fixtures'
    rows = []
    for app in apps:
        for bits in ([64, 32] if app == 'CHT' else [64]):
            tests = [(name, fixture / 'shared_operators' / (name + '.py'), []) for name in ['source_balance', 'collision_split_behavior', 'operator_contract', 'extended_operators', 'moment_publication']]
            tests.append(('tracking_grid', fixture / 'shared_operators' / 'tracking_grid.py', ['--require-independent', '--require-particle-independent']))
            tests.append(('pressure_split', fixture / 'pressure' / 'limited_flux_balance.py', []))
            if app == 'gasUGKP':
                tests.append(('pressure_unsorted', fixture / 'pressure' / 'limited_flux_balance.py', ['unsorted']))
            else:
                tests.extend((name, fixture / 'shared_operators' / (name + '.py'), []) for name in ['payload_commit', 'contact_conservation', 'contact_age_theta'])
            for name, script, extra in tests:
                label = f'{app}{bits}-{name}'
                log = output / (label + '.log')
                command = [sys.executable, str(script), str(root), str(output / label), app, str(bits), *extra]
                with log.open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                rows.append(dict(test=name, app=app, bits=bits, exit=result.returncode, log=str(log)))
                (output / 'results.json').write_text(json.dumps(rows, indent=2))
                print(label, 'PASS' if result.returncode == 0 else f'FAIL ({result.returncode})', flush=True)
                if result.returncode:
                    print(log.read_text()[-4000:])
                    return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
