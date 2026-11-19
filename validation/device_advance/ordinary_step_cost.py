"""Measure warm FP32 CHT step cost on the eight-cell clock fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from cht_physical_clock import replace_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'fixture', 'output'):
        parser.add_argument(name, type=lambda s: Path(s).resolve())
    parser.add_argument('--particle-restart', type=Path, help='two-parcel native restart for this mesh')
    parser.add_argument('--mutable', action='store_true')
    parser.add_argument('--capacity', type=int)
    args = parser.parse_args()
    points = (args.fixture/'constant/polyMesh/points').read_text()
    if not re.search(r'\n27\s*\n\(', points):
        parser.error('expected an eight-cell fixture')
    case = args.output
    case.mkdir(parents=True)
    for name in ('0', 'constant', 'system'):
        shutil.copytree(args.fixture/name, case/name)
    ctl = case/'system/controlDict'
    scheme = case/'constant/scheme'

    dt = 2**-24
    block = 64 if args.mutable else 512
    for key, value in dict(deltaT=dt, endTime=dt*block*8, logInterval=block,
                           adjustTimeStep='yes', runTimeModifiable='true' if args.mutable else 'false',
                           writeInterval=-1, maxThermalCouplingEvents=0).items():
        replace_entry(ctl, key, value)
    for key, value in dict(precision=32, maxDeltaT=dt, courantUpdateInterval=128,
                           couplingInterval=1, solidContactCouplingInterval=1).items():
        replace_entry(scheme, key, value)
    if args.particle_restart:
        shutil.copyfile(args.particle_restart, case/'0/gpuResidentStrictParticles.dat')
        replace_entry(case/'constant/schedulingProperties', 'gpuCsrLevel', 'L1')
    if args.capacity is not None:
        if args.capacity < 2:
            parser.error('capacity must accommodate the two-parcel fixture')
        replace_entry(case/'constant/schedulingProperties', 'gpuResidentParticleCapacity', args.capacity)
    times = {}
    started = time.perf_counter()
    with (case/'solver.log').open('x') as log, subprocess.Popen(
        [str(args.binary), '-case', str(case)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ) as proc:
        for line in proc.stdout:
            log.write(line)
            match = re.match(r'^UGKP_REPRO step=(\d+) time=(\S+) dt=(\S+)', line)
            if match:
                step = int(match[1])
                assert float(match[2]) == step*dt and float(match[3]) == dt, line
                times[step] = time.perf_counter()-started
        if proc.wait():
            raise RuntimeError(f'solver failed, see {log.name}')
    if set(times) != set(range(block, 8*block+1, block)):
        raise ValueError(f'incomplete progress samples: {times}')
    samples = [(times[i+block]-times[i])*1e6/block for i in range(block, 7*block, block)]
    result = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  cells=8, particle_restart=bool(args.particle_restart), mutable=args.mutable, timed_steps=6*block,
                  warm_us_per_step=sum(samples)/len(samples), intervals_us_per_step=samples,
                  first_log_s=times[block], elapsed_s=time.perf_counter()-started)
    if args.capacity is not None:
        result['particle_capacity'] = args.capacity
    counters = re.search(r'^ADVANCE_BATCHES count=(\d+) steps=(\d+) single=(\d+) max=(\d+)',
                         (case/'solver.log').read_text(), re.M)
    if counters:
        batches,steps,single,longest = map(int,counters.groups())
        assert steps == block*8 and longest == (1 if args.mutable else block)
        result['batches'] = dict(count=batches, steps=steps, single=single, maximum=longest)
    (case/'timing.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
