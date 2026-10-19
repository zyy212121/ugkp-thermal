#!/usr/bin/env python3
"""Reproduce the CUDA wall replay against any compatible source checkout.

Writes only below --out. The input fixtures and capture declaration are taken
from this evidence checkout, while solver headers come from --source first.
"""
import argparse, hashlib, json, pathlib, statistics, subprocess

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=pathlib.Path)
    p.add_argument('--out', required=True, type=pathlib.Path)
    p.add_argument('--cuda', type=pathlib.Path, default=pathlib.Path('/usr/local/cuda-13.1'))
    p.add_argument('--arch', default='sm_89')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--sanitizers', action='store_true')
    a = p.parse_args()
    bundle = pathlib.Path(__file__).resolve().parent.parent
    repository = bundle.parents[2]
    a.out.mkdir(parents=True, exist_ok=True)
    commands = []
    def run(command, name):
        q = subprocess.run(list(map(str, command)), capture_output=True, text=True)
        (a.out / (name + '.log')).write_text(q.stdout + q.stderr)
        commands.append({'command': list(map(str, command)), 'exit': q.returncode})
        (a.out / 'commands.json').write_text(json.dumps(commands, indent=2))
        q.check_returncode()
        return q.stdout
    compile_flags = [a.cuda/'bin/nvcc', '-std=c++17', '-O3', '-lineinfo', '--fmad=true', '-arch='+a.arch, '-Xptxas=-v', '-I'+str(a.source.resolve()), '-I'+str(repository)]
    run(compile_flags + [bundle/'tools/wall_replay.cu', '-o', a.out/'replay'], 'build')
    for r in range(a.rounds):
        for label, filename in [('homogeneous', 'captured004-homogeneous336.csv'), ('heterogeneous', 'actual-wall-input-first-restart-step.csv')]:
            print(run([a.out/'replay', bundle/'inputs'/filename, a.out/f'{label}-{r}.csv'], f'{label}-{r}'), flush=True)
    run([a.cuda/'bin/cuobjdump', '-res-usage', a.out/'replay'], 'resources')
    if a.sanitizers:
        fixture = repository/'validation/tests/gas_wall/fixtures/wall_workspace_pool_gpu.cu'
        run(compile_flags + [fixture, '-o', a.out/'pool'], 'pool-build')
        for tool in ['memcheck', 'racecheck', 'synccheck']:
            run([a.cuda/'bin/compute-sanitizer', '--tool', tool, '--error-exitcode', '99', a.out/'pool'], 'pool-'+tool)
    hashes = {x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.out.iterdir() if x.is_file()}
    (a.out/'sha256.json').write_text(json.dumps(hashes, indent=2))

if __name__ == '__main__':
    main()
