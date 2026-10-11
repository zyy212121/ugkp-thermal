#!/usr/bin/env python3
"""Label a recorded bounded process sample; never launch or authorize a run."""
import argparse
import json
import math
from pathlib import Path


def estimate(sample):
    keys=('physical_start_s','physical_end_s','target_end_s','wall_seconds')
    try:start,end,target,wall=(float(sample[key]) for key in keys)
    except (KeyError,TypeError,ValueError):raise ValueError('sample requires finite physical times and wall_seconds') from None
    if not all(math.isfinite(value) for value in (start,end,target,wall)) or start<0 or end<=start or target<end or wall<=0:
        raise ValueError('invalid progress or duration in performance sample')
    for key in ('concurrent','instrumented'):
        if key in sample and sample[key] is not None and type(sample[key]) is not bool:
            raise ValueError(key+' must be a boolean or unverified null')
    limitations=[]
    if end<target:limitations.append('PARTIAL_TRAJECTORY')
    for key,active,unknown in [('concurrent','CONCURRENT_WORK','CONCURRENCY_UNVERIFIED'),
                              ('instrumented','INSTRUMENTED_PROCESS','INSTRUMENTATION_UNVERIFIED')]:
        if sample.get(key) is None:limitations.append(unknown)
        elif sample[key]:limitations.append(active)
    remaining=(target-end)/(end-start)*wall
    total=(target-start)/(end-start)*wall
    if not math.isfinite(remaining) or not math.isfinite(total):raise ValueError('performance extrapolation overflow')
    return {'status':'DIAGNOSTIC_LINEAR_EXTRAPOLATION' if end<target else 'COMPLETED_SAMPLE',
        'sample':dict(sample),'limitations':limitations,
        'estimated_remaining_wall_seconds':remaining,'estimated_total_from_sample_start_wall_seconds':total,
        'timing_scope':'PROCESS_WALL_TIME_INCLUDING_STARTUP_AND_IO',
        'isolated_gpu_throughput':'NOT_MEASURED','eta_reliability':'NOT_ESTABLISHED',
        'authorizes_full_run':False,
        'assumption':'Linear extrapolation assumes unchanged timestep, acceptance rate, workload and hardware contention; this sample does not establish those assumptions.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sample',type=Path);parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    if arguments.output.exists():raise FileExistsError('performance report output must be fresh')
    result=estimate(json.loads(arguments.sample.read_text()))
    arguments.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
