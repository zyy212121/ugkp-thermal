import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from cht_physical_clock import number, replace_entry


def main():
    parser=argparse.ArgumentParser(description='Change an active eight-cell run through its OpenFOAM control dictionary.')
    for name in ('binary','fixture','output'):
        parser.add_argument(name,type=lambda p:Path(p).resolve())
    parser.add_argument("--enable-adaptive",action="store_true")
    args=parser.parse_args()
    if not re.search(r'\bnCells:\s*8\b',(args.fixture/'constant/polyMesh/owner').read_text()):
        parser.error('expected an eight-cell fixture')
    args.output.mkdir(parents=True)
    for name in ('0','constant','system'):
        shutil.copytree(args.fixture/name,args.output/name)
    control=args.output/'system/controlDict'
    dt=2**-24
    original_steps=8192
    for key,value in dict(endTime=dt*original_steps,deltaT=dt,writeInterval=original_steps,
                          logInterval=1,runTimeModifiable='true',adjustTimeStep='false').items():
        replace_entry(control,key,value)
    if args.enable_adaptive:
        scheme=args.output/'constant/scheme'
        replace_entry(scheme,'maxDeltaT',4*dt)
        replace_entry(scheme,'courantUpdateInterval',1)
    earlier=time.time()-60
    os.utime(control,(earlier,earlier))
    target=0
    observed=[]
    physical=[]
    env=dict(os.environ,UGKP_ADVANCE_STATS='1')
    with (args.output/'solver.log').open('x') as log,subprocess.Popen(
            [args.binary,'-case',args.output],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env) as proc:
        for line in proc.stdout:
            log.write(line)
            match=re.match(r'^UGKP_REPRO step=(\d+) time=(\S+) dt=(\S+)',line)
            if not match:
                continue
            step=int(match[1]);observed.append(step)
            t,step_dt=float(match[2]),float(match[3])
            physical.append((t,step_dt))
            if not args.enable_adaptive and (t!=step*dt or step_dt!=dt):
                raise ValueError(f'Published a different physical step: {line}')
            if step>=64 and not target:
                target=step+512
                pending=control.with_suffix('.new')
                shutil.copyfile(control,pending)
                replace_entry(pending,'endTime',target*dt)
                if args.enable_adaptive:
                    replace_entry(pending,'adjustTimeStep','true')
                    replace_entry(pending,'writeControl','adjustableRunTime')
                    replace_entry(pending,'writeInterval',target*dt/2)
                else:
                    replace_entry(pending,'writeInterval',target//2)
                os.replace(pending,control)
        if proc.wait():
            raise RuntimeError(f'Failed run in {args.output}')
    if not target or observed!=list(range(1,len(observed)+1)) or physical[-1][0]!=target*dt:
        raise ValueError(f'Changed end time was missed, expected {target*dt}, last records {physical[-2:]}')
    if args.enable_adaptive:
        if len(observed)>=target or not any(dt<v<=4*dt for t,v in physical) or max(v for t,v in physical)>4*dt:
            raise ValueError('Runtime switch did not enable step growth within the configured maximum')
    elif len(observed)!=target:
        raise ValueError('Fixed step count changed unexpectedly')
    times={float(p.name):p for p in args.output.iterdir() if p.is_dir() and re.fullmatch(r'[+\d.eE-]+',p.name)}
    for time_value in (target*dt/2,target*dt):
        if time_value not in times:
            raise ValueError(f'Changed write cadence missed {time_value}')
        index=int(number(times[time_value]/'uniform/time','index'))
        if physical[index-1][0]!=time_value:
            raise ValueError(f'Write does not match accepted clock at {time_value}')
    stats=re.search(r'ADVANCE_BATCHES count=(\d+) steps=(\d+) single=(\d+) max=(\d+)',(args.output/'solver.log').read_text())
    if not stats or int(stats[4])<=1 or int(stats[2])!=len(observed):
        raise ValueError('Did not exercise observed multi-step submissions')
    print(f'Accepted {len(observed)} steps with writes at {target*dt/2} and {target*dt}. {stats[0]}')


if __name__=='__main__':
    main()
