#!/usr/bin/env python3
"""Profile real interface runs with Nsight; includes startup and normal output.

Requires sourced OpenFOAM 10, CUDA and Nsight Systems. No production parameters
are changed. The full-run transfer counts are structural evidence, not timings
for an unprofiled production build.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('matrix', ROOT/'validation/molecular_transport/native_matrix.py')
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def summarize(path):
    with sqlite3.connect(path) as db:
        copies = db.execute('''SELECT e.label, COUNT(*), SUM(m.bytes), SUM(m.end-m.start)
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind=e.id GROUP BY m.copyKind''').fetchall()
        runtime = db.execute('''SELECT s.value, COUNT(*), SUM(r.end-r.start)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
            GROUP BY r.nameId''').fetchall()
    return {
        'copies': {name: {'count': count, 'bytes': size, 'duration_ns': ns}
                   for name, count, size, ns in copies},
        'runtime': {name: {'count': count, 'duration_ns': ns}
                    for name, count, ns in runtime},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['chmt-mixture', 'cht-contact'])
    p.add_argument('--nsys', default='/usr/local/cuda/bin/nsys')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    models = json.loads((matrix.HERE/'fixtures.json').read_text())
    result = {'scope': 'whole native run including initialization and normal output', 'cases': {}}
    for name in a.cases:
        result['cases'][name] = {}
        for kind, bindir in [('baseline', a.baseline), ('candidate', a.candidate)]:
            case = a.work/f'{name}-{kind}'
            meta = models[name]
            matrix.prepare(matrix.HERE/'fixtures'/name, case, meta)
            prefix = str(case)
            command = [a.nsys, 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
                       '--force-overwrite=true', '-o', prefix,
                       str(bindir.resolve()/meta['solver']), '-case', str(case.resolve())]
            matrix.run(command, a.work/f'{name}-{kind}.profile.log')
            with (a.work/f'{name}-{kind}.export.log').open('w') as log:
                subprocess.run([a.nsys, 'export', '--type=sqlite', '--force-overwrite=true',
                                '-o', prefix+'.sqlite', prefix+'.nsys-rep'],
                               env=matrix.ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
            result['cases'][name][kind] = summarize(prefix+'.sqlite')
            (a.work/'results.json').write_text(json.dumps(result, indent=2)+'\n')
            print(name, kind, result['cases'][name][kind]['copies'], flush=True)


if __name__ == '__main__':
    main()
