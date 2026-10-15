#!/usr/bin/env python3
"""Inspect a saved correctness bundle. Execution requires explicit --run.

Source and binary hashes are offline audit data, not execution admission.
Expected stdout remains a post-execution correctness result, with no timing.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--run', action='store_true', help='Execute the bundle correctness programs')
args = parser.parse_args()
bundle = args.bundle.resolve()
manifest = json.loads((bundle / 'deferred-run.json').read_text())
programs = []
for row in manifest['programs']:
    path = (bundle / row['executable']).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        sys.exit('Executable outside bundle or missing: ' + row['executable'])
    programs.append((row, path))
print('Inspected', len(programs), 'bundle programs', flush=True)
if not args.run:
    print('Inspection only. GPU execution remains pending; use --run during a correctness window.')
    sys.exit(0)
logs = bundle / 'gpu-correctness'
logs.mkdir(exist_ok=True)
results = []
for row, path in programs:
    for index, arguments in enumerate(row['arguments']):
        result = subprocess.run([str(path), *arguments], capture_output=True, text=True, timeout=600)
        label = row['test'] + '-' + str(index)
        (logs / (label + '.log')).write_text(result.stdout + result.stderr)
        ok = result.returncode == 0 and row['expected_stdout'] in result.stdout
        results.append(dict(test=row['test'], arguments=arguments, exit=result.returncode, passed=ok))
        (logs / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(label, 'PASS' if ok else 'FAIL', flush=True)
        if not ok:
            print(result.stdout + result.stderr)
            sys.exit(1)
print('GPU correctness complete:', len(results), 'checks. No performance timing.', flush=True)
