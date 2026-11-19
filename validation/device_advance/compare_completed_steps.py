import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import time

import numpy as np
from analyze import dictionary, field, mesh


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_files(case):
    return sorted(p for folder in ('0','system','constant') for p in (case/folder).rglob('*') if p.is_file())


def input_digest(case):
    h=hashlib.sha256()
    for path in case_files(case):
        h.update(str(path.relative_to(case)).encode()+b'\0'+bytes.fromhex(digest(path)))
    return h.hexdigest()


def totals(values,weights):
    values=np.asarray(values).reshape(len(weights),-1)
    if not len(weights) or not np.isfinite(values).all():
        raise ValueError('Empty or nonfinite physical field')
    return dict(mean=np.average(values,axis=0,weights=weights).tolist(),
                mean_abs=np.average(np.abs(values),axis=0,weights=weights).tolist())


def foam_results(case,time,geometry):
    terminal,=[p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[+\d.eE-]+',p.name) and float(p.name)==time]
    weights=geometry['volume']
    names=sorted(p.name for p in terminal.iterdir() if p.is_file() and
                 (p.name in ('rho','rhoE','rhoU','U','p','T','k','omega','nut','rhoUs','rhoDs','rhoHp') or p.name.startswith('Y_')))
    if not {'rho','rhoE','rhoU'}<=set(names):
        raise ValueError(f'Missing conserved output at {terminal}')
    result={name:totals(field(terminal/name,len(weights)),weights) for name in names}
    if not (terminal/'thermalExchangeState').is_file():
        return result,{}
    stamp=dictionary(terminal/'thermalExchangeState')
    clocks={k:stamp[k] for k in ('exchangeSequence','completedSimulationTimeS','completedRadiationSimulationTimeS')}
    solid_volume=mesh(case/'constant/solid/polyMesh')['volume']
    result['solid/T']=totals(field(terminal/'solid/T',len(solid_volume)),solid_volume)
    for patch,(start,count) in geometry['patches'].items():
        if not count:
            continue
        area=geometry['area'][start:start+count]
        total=np.zeros(count)
        for name in ('gasConvectiveWallHeatFlux','particleRadiationWallHeatFlux','particleReflectedWallHeatFlux','particleStuckWallHeatFlux'):
            values=field(terminal/name,count,patch)
            total+=values
            result[f'{patch}/{name}']=totals(values,area)
        result[f'{patch}/totalWallHeatFlux']=totals(total,area)
    return result,clocks


def radiation_flux_resolution(case,time):
    for saved in case.glob('*/domWarmStart.bin'):
        if not re.fullmatch(r'[+\d.eE-]+',saved.parent.name) or float(saved.parent.name)!=time:
            continue
        with saved.open('rb') as source:
            magic,version,width,cells,angles,layout,warm=struct.unpack('=7Q',source.read(56))
            if magic!=0x55474b50444f4d31 or width not in (4,8) or warm not in (0,1):
                raise ValueError(f'Invalid angular field in {saved}')
            if not warm:
                return 0.
            dtype=np.float32 if width==4 else np.float64
            angular=np.fromfile(source,dtype=dtype)
            if angular.size!=cells*angles or not np.isfinite(angular).all():
                raise ValueError(f'Incomplete angular field in {saved}')
            gross=dtype(4*math.pi*float(np.max(np.abs(angular),initial=0)))
            return float(np.spacing(gross))
    return 0.


def chmt_results(case,step):
    directory=case/'checkpointOutput'
    def rows(name):
        with (directory/f'{name}-{step}.csv').open() as stream:
            return list(csv.DictReader(stream))
    stamp,=rows('accepted')
    result={}
    for table in ('gas','material','exchange'):
        data=rows(table)
        if not data:
            continue
        for column in data[0]:
            if column in ('cell','packet','step','geometry','face','stage','kind','gas_cell','solid_cell','consumer_mask'):
                continue
            values=[float(r[column]) for r in data]
            if not all(map(math.isfinite,values)):
                raise ValueError(f'Nonfinite {table}/{column}')
            result[f'{table}/{column}']=dict(mean=[math.fsum(values)],mean_abs=[math.fsum(map(abs,values))])
    particle_file=directory/f'particles-{step}.csv'
    if particle_file.is_file():
        particles=rows('particles')
        for key in ('mass','thermal_energy','thermal_carry','contact_mechanical_energy'):
            amounts=[float(row[key]) for row in particles]
            if not all(map(math.isfinite,amounts)):
                raise ValueError(f'Nonfinite particle {key}')
            result['particles/'+key]=dict(mean=[math.fsum(amounts)],mean_abs=[math.fsum(map(abs,amounts))])
        result['particles/count']=dict(mean=[len(particles)],mean_abs=[len(particles)])
    return result,{k:stamp[k] for k in ('accepted_steps','gas_cells','material_cells','interface_faces','commit_sequence')}


def run(binary,fixture,probe,case,geometry,fingerprint,blocks):
    case.mkdir()
    for name in ('0','constant','system'):
        shutil.copytree(fixture/name,case/name)
    if input_digest(case)!=fingerprint:
        raise RuntimeError('Case inputs changed during copy')
    first=max(2,blocks//4)
    last=blocks-max(1,blocks//8)
    env=dict(os.environ,LD_PRELOAD=str(probe),UGKP_COMPLETED_PROGRESS=str(case/'completed.csv'),
             UGKP_PROGRESS_FIRST=str(first),UGKP_PROGRESS_LAST=str(last))
    launch=time.monotonic_ns()
    with (case/'solver.log').open('x') as log:
        status=subprocess.run([binary,'-case',case],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
    process_seconds=(time.monotonic_ns()-launch)/1e9
    log=(case/'solver.log').read_text()
    if status:
        raise RuntimeError(f'{binary.name} failed in {case}\n{log[-1500:]}')
    progress=[(int(n),float(t),float(dt)) for n,t,dt in re.findall(
        r'^UGKP_REPRO step=(\d+) time=(\S+) dt=(\S+)',log,re.M)]
    microsteps=[]
    if not progress:
        windows=re.findall(r'^CHMT_REPRO step=(\d+) time=(\S+).*?gasMicrosteps=(\d+) materialSolves=(\d+)',log,re.M)
        progress=[(int(i),float(t),0.0) for i,t,g,m in windows]
        microsteps=[(int(g),int(m)) for i,t,g,m in windows]
    stamps=[tuple(map(int,line.split(','))) for line in (case/'completed.csv').read_text().splitlines()]
    if len(progress)!=blocks or [r[0] for r in stamps]!=list(range(1,blocks+1)):
        raise ValueError(f'Unexpected progress boundaries in {case}: {len(progress)}, {len(stamps)}')
    if [i for i,a,b in stamps if b]!=[first,last] or any(a>b for _,a,b in stamps if b):
        raise ValueError('Missing GPU completion at the measurement boundaries')
    work=[b[0]-a[0] for a,b in zip(progress,progress[1:])]
    if min(work)<1 or len(set(work))!=1:
        raise ValueError('Progress intervals represent different numbers of steps')
    fields,clock=chmt_results(case,progress[-1][0]) if microsteps else foam_results(case,progress[-1][1],geometry)
    settings={'.'.join(v for v in (region,module,key) if v):value
              for module,key,region,value in re.findall(
                  r'^SCHEME_VALUE module=(\S+) key=(\S+)(?: region=(\S+))? effective=(\S+)',log,re.M)}
    precision=re.search(r'^SCHEME_STARTUP .*?\bprecision=(32|64)\b',log,re.M)
    if not precision:
        raise ValueError(f'Missing compiled precision in {case}')
    settings['precision']=int(precision[1])
    turbulence=re.search(r'^Gas turbulence:\s*(.*)$',log,re.M)
    settings['turbulence']=turbulence[1] if turbulence else None
    adaptive=re.search(r'^(?:CHMT_)?TIME_CONTROL adjustTimeStep=(\d+)',log,re.M)
    if adaptive:
        settings['adaptive']=int(adaptive[1])
    start,stop=first-1,last-1
    timed_steps=progress[stop][0]-progress[start][0]
    return dict(binary=digest(binary),progress=progress,steps=timed_steps,
                us_per_step=(stamps[stop][2]-stamps[start][2])/(1000*timed_steps),
                host_interval_us=[(stamps[i+1][1]-stamps[i][1])/(1000*work[i]) for i in range(start,stop)],
                boundary_drain_us=[(b-a)/1000 for _,a,b in stamps if b],process_seconds=process_seconds,
                startup_to_first_progress_s=(stamps[0][1]-launch)/1e9,
                final_progress_to_exit_s=process_seconds-(stamps[-1][1]-launch)/1e9,
                radiation_flux_resolution=0. if microsteps else radiation_flux_resolution(case,progress[-1][1]),
                fields=fields,work=microsteps,accepted_clock=clock,settings=settings)


def assess(trials,fingerprint,cells):
    baseline=trials[0]
    errors={}
    absolute={}
    within_roundoff=set()
    passed_physics=True
    for trial in trials:
        if any(trial[k]!=baseline[k] for k in ('progress','accepted_clock','work')) or trial['fields'].keys()!=baseline['fields'].keys():
            raise RuntimeError('Physical steps, times or published fields differ')
        for key in trial['settings'].keys() & baseline['settings'].keys():
            if trial['settings'][key]!=baseline['settings'][key]:
                raise RuntimeError(f'Effective solver settings differ for {key}')
        for name,field in trial['fields'].items():
            ref=baseline['fields'][name]
            if field.keys()!=ref.keys() or any(len(field[key])!=len(ref[key]) for key in field):
                raise RuntimeError(f'Published field components differ for {name}')
            scale=max(ref['mean_abs']+field['mean_abs']+[1e-30])
            difference=max(abs(a-b) for key in field for a,b in zip(ref[key],field[key]))
            relative=difference/scale
            errors[name]=max(errors.get(name,0),relative)
            absolute[name]=max(absolute.get(name,0),difference)
            if relative>2e-5:
                resolution=max(trial['radiation_flux_resolution'],baseline['radiation_flux_resolution'])
                if name.endswith('/particleRadiationWallHeatFlux') and difference<=resolution:
                    within_roundoff.add(name)
                else:
                    passed_physics=False
    perf={key:statistics.mean(t['us_per_step'] for t in trials if t['role']==key) for key in ('reference','candidate')}
    by_pair=[next(t['us_per_step'] for t in trials if t['pair']==i and t['role']=='candidate')/
             next(t['us_per_step'] for t in trials if t['pair']==i and t['role']=='reference') for i in range(1,len(trials)//2+1)]
    timing='no_slower_in_all_pairs' if max(by_pair)<=1 else 'slower_in_all_pairs' if min(by_pair)>1 else 'order_dependent'
    return dict(timing=timing,cells=cells,input_sha256=fingerprint,reference_us=perf['reference'],candidate_us=perf['candidate'],
                time_ratio=perf['candidate']/perf['reference'],paired_ratios=by_pair,
                macro_relative_errors=errors,macro_absolute_errors=absolute,within_radiation_roundoff=sorted(within_roundoff),
                physics_match=passed_physics,trials=trials)


def main():
    parser=argparse.ArgumentParser(description='Pair ordinary transport runs at completed GPU progress boundaries on a small case.')
    for name in ('reference','candidate','fixture','probe','output'):
        parser.add_argument(name,type=lambda p:Path(p).resolve())
    parser.add_argument('--pairs',type=int,default=2,choices=(1,2,4))
    parser.add_argument('--blocks',type=int,default=8)
    args=parser.parse_args()
    if args.blocks<5:
        parser.error('at least five progress records are required')
    topology=(args.fixture/'constant/polyMesh/owner').read_text()
    cells=int(re.search(r'\bnCells:\s*(\d+)',topology)[1])
    if cells>4096:
        parser.error('this benchmark accepts a small mesh with at most 4096 cells')
    geometry=mesh(args.fixture/'constant/polyMesh')
    args.output.mkdir(parents=True)
    binaries=dict(reference=args.reference,candidate=args.candidate)
    fingerprint=input_digest(args.fixture)
    trials=[]
    for pair in range(args.pairs):
        order=('reference','candidate') if pair%2==0 else ('candidate','reference')
        for role in order:
            trial=run(binaries[role],args.fixture,args.probe,args.output/f'{pair+1}-{role}',geometry,fingerprint,args.blocks)
            trial.update(pair=pair+1,role=role)
            trials.append(trial)
            print(f"{args.fixture.name} {role} {trial['us_per_step']:.3f} us/step",flush=True)
    result=assess(trials,fingerprint,cells)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','macro_relative_errors','macro_absolute_errors')}))
    if not result['physics_match']:
        raise SystemExit('Physics comparison failed')


if __name__=='__main__':
    main()
