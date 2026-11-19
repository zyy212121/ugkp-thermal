"""Exercise CHT's transient clock on a prepared eight-cell radiation fixture."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def replace_entry(path, key, value):
    text = path.read_text()
    expression = r'(?m)^(\s*' + key + r'\s+)[^;]+;'
    text, count = re.subn(expression, lambda m: m[1] + str(value) + ';', text)
    if count == 0:
        text += f'\n{key} {value};\n'
    elif count != 1:
        raise ValueError(f'{path}: ambiguous {key}')
    path.write_text(text)


def number(path, key):
    match = re.search(r'\b' + key + r'\s+"?([-+.\deE]+)"?;', path.read_text())
    return float(match[1])


def simulate(args, name, binary, *, adaptive, dt, end, period, write, frequency=2, restart=None, log_interval=1, write_control=None, event_cap=0):
    case = args.output / name
    case.mkdir()
    for directory in ('0','constant','system'):
        shutil.copytree(args.fixture/directory, case/directory)
    if restart:
        shutil.copytree(restart,case/restart.name)
    ctl = case/'system/controlDict'
    scheme = case/'constant/scheme'
    for key,value in dict(startTime=restart.name if restart else 0,stopAt='endTime',
                          deltaT=dt,endTime=end,writeInterval=write,logInterval=log_interval,
                          maxThermalCouplingEvents=event_cap,adjustTimeStep='yes' if adaptive else 'no').items():
        replace_entry(ctl,key,value)
    if write_control:
        replace_entry(ctl,'writeControl',write_control)
    for key,value in dict(precision=args.precision,couplingInterval=period,solidContactCouplingInterval=period,
                          radiationCouplingFrequency=frequency,maxDeltaT=8e-6,
                          maximumTimeStepGrowth=2,courantUpdateInterval=3).items():
        replace_entry(scheme,key,value)
    if frequency==0:
        rad=case/'constant/radiationProperties'
        replace_entry(rad,'enabled','false')
        solid=case/'constant/solidRegionProperties'
        solid.write_text(solid.read_text().replace('radiationEnabled true','radiationEnabled false'))
    log = case/'run.log'
    with log.open('x') as out:
        rc = subprocess.run([str(binary), '-case', str(case)], stdout=out, stderr=subprocess.STDOUT).returncode
    text=log.read_text()
    assert rc==0,text[-1800:]
    progress=[(int(i),float(t),float(h)) for i,t,h in re.findall(
        r'^UGKP_REPRO step=(\d+) time=(\S+) dt=(\S+)',text,re.M)]
    done=re.search(r'^THERMAL_RUN_COMPLETE heatEvents=(\d+) radiationEvents=(\d+) time=(\S+)',text,re.M)
    assert done,text[-1800:]
    snapshots={}
    for marker in case.glob('*/thermalExchangeState'):
        t=float(marker.parent.name)
        if t==0 or (restart and t==float(restart.name)):
            continue
        ledger=marker.parent/'gpuPendingWallEnergyLedgers.dat'
        state=dict(sequence=number(marker,'exchangeSequence'),heat=number(marker,'completedSimulationTimeS'),
                   radiation=number(marker,'completedRadiationSimulationTimeS'),
                   last_exchange=number(marker,'previousExchangeSimulationTimeS'),
                   checkpoint=number(ledger,'checkpointSimulationTimeS'),directory=marker.parent.name)
        assert state['checkpoint']==t and number(marker.parent/'uniform/time','value')==t
        assert (marker.parent/'gpuResidentStrictParticles.dat').is_file()
        if name!='archived-adaptive':
            assert state['heat']==t,(name,t,state)
            if frequency and (write!=-1 or t==end):
                assert state['radiation']==t,(name,t,state)
        snapshots[t]=state
    terminal=max(snapshots)
    final_index=number(case/snapshots[terminal]['directory']/'uniform/time','index')
    initial_index=number(restart/'uniform/time','index') if restart else 0
    return dict(final=float(done[3]),heat_events=int(done[1]),radiation_events=int(done[2]),
                steps=int(final_index-initial_index),first_dt=progress[0][2] if progress else None,
                observed_dt=sorted(set(h for _,_,h in progress)),snapshots=snapshots)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('binary','fixture','output'):
        parser.add_argument(arg,type=lambda p:Path(p).resolve())
    parser.add_argument('--precision',type=int,choices=(32,64),default=32)
    parser.add_argument('--before-fix',type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)

    points=(args.fixture/'constant/polyMesh/points').read_text()
    assert re.search(r'\n27\s*\n\(',points) and re.search(r'\benabled\s+(?:true|on|yes|1)\s*;', (args.fixture/'constant/radiationProperties').read_text())
    cases={}
    options=dict(adaptive=True,dt=1e-6,end=4e-5,period=2e-5,write=-1)
    if args.before_fix:
        old=simulate(args,'archived-adaptive',args.before_fix,**options)
        assert old['observed_dt']==[1e-6]
        cases['archived-adaptive']=old
    fixed=simulate(args,'adaptive',args.binary,**options)
    assert fixed['first_dt']==2e-6 and max(fixed['observed_dt'])<=8e-6 and max(fixed['observed_dt'])>fixed['first_dt']
    assert fixed['final']==4e-5 and (fixed['heat_events'],fixed['radiation_events'])==(2,1)
    assert set(fixed['snapshots'])=={2e-5,4e-5}
    assert fixed['snapshots'][4e-5]['radiation']==4e-5
    if args.before_fix:
        assert fixed['steps']<old['steps']
    cases['adaptive']=fixed
    adaptive_checkpoint=args.output/'adaptive'/fixed['snapshots'][2e-5]['directory']
    assert fixed['snapshots'][2e-5]['radiation']==0
    cold=(adaptive_checkpoint/'domWarmStart.bin').read_bytes()
    warm=(args.output/'adaptive'/fixed['snapshots'][4e-5]['directory']/'domWarmStart.bin').read_bytes()
    assert len(cold)==56 and int.from_bytes(cold[48:56],'little')==0
    assert len(warm)>56 and int.from_bytes(warm[48:56],'little')==1
    actual=number(adaptive_checkpoint/'uniform/time','deltaT')
    nominal=number(adaptive_checkpoint/'uniform/time','cflDeltaT')
    assert 0<actual<nominal and nominal==8e-6
    warm=simulate(args,'adaptive-restart',args.binary,**options,restart=adaptive_checkpoint)
    assert warm['first_dt']==nominal and warm['final']==4e-5
    assert (warm['heat_events'],warm['radiation_events'])==(1,1)
    cases['adaptive-restart']=warm
    capped=simulate(args,'event-cap',args.binary,**options,event_cap=1)
    assert (capped['heat_events'],capped['radiation_events'])==(1,1) and capped['final']==2e-5
    assert capped['snapshots'][2e-5]['radiation']==2e-5
    cases['event-cap']=capped

    offset=simulate(args,'offset-output',args.binary,adaptive=False,dt=7e-6,end=4.5e-5,period=2e-5,write=2.1e-5)
    assert set(offset['snapshots'])=={2.1e-5,4.2e-5,4.5e-5} and offset['first_dt']==7e-6
    assert [(offset['snapshots'][t]['sequence'],offset['snapshots'][t]['heat']) for t in sorted(offset['snapshots'])]==[(2,2.1e-5),(4,4.2e-5),(5,4.5e-5)]
    assert (offset['heat_events'],offset['radiation_events'])==(5,4)
    cases['offset-output']=offset

    near_end=args.output/'offset-output'/offset['snapshots'][4.2e-5]['directory']
    short=simulate(args,'short-final-restart',args.binary,adaptive=False,dt=7e-6,end=4.5e-5,
                   period=2e-5,write=2.1e-5,restart=near_end)
    assert short['steps']==1 and set(short['snapshots'])=={4.5e-5}
    assert (short['heat_events'],short['radiation_events'])==(1,1)
    assert short['snapshots'][4.5e-5]['heat']==4.5e-5
    cases['short-final-restart']=short

    every=simulate(args,'heat-every-step',args.binary,adaptive=False,dt=7e-6,end=2.1e-5,
                   period=0,frequency=0,write=-1)
    assert every['heat_events']==every['steps']==3 and every['radiation_events']==0
    assert len(every['snapshots'])==3 and every['final']==2.1e-5
    cases['heat-every-step']=every

    aligned=simulate(args,'aligned',args.binary,adaptive=False,dt=7e-5,end=0.01,period=.005,write=.005)
    assert set(aligned['snapshots'])=={.005,.01} and aligned['final']==.01
    assert [(aligned['snapshots'][t]['heat'],aligned['snapshots'][t]['radiation']) for t in (.005,.01)]==[(.005,.005),(.01,.01)]
    cases['aligned']=aligned
    grouped=simulate(args,'grouped',args.binary,adaptive=False,dt=7e-5,end=.01001,period=.005,
                     write=.01001,log_interval=5000)
    assert set(grouped['snapshots'])=={.01001}
    assert (grouped['heat_events'],grouped['radiation_events'])==(3,2)
    assert grouped['snapshots'][.01001]['last_exchange']==.01 and grouped['snapshots'][.01001]['heat']==.01001
    event=re.search(r'^THERMAL_EVENT accepted=3 radiation=2 time=(\S+)',(args.output/'grouped/run.log').read_text(),re.M)
    assert event and float(event[1])==.01001
    cases['grouped']=grouped



    starts=args.output/'aligned'/aligned['snapshots'][.005]['directory']
    phase=simulate(args,'resumed-phase',args.binary,adaptive=False,dt=7e-5,end=.012,period=.005,
                   write=.012,restart=starts,log_interval=5000)
    assert (phase['heat_events'],phase['radiation_events'])==(2,2)
    cases['resumed-phase']=phase

    indexed=simulate(args,'step-output',args.binary,adaptive=False,dt=1e-5,end=7e-5,period=3e-5,
                     write=2,write_control='timeStep',log_interval=5000)
    assert set(indexed['snapshots'])=={2e-5,4e-5,6e-5,7e-5}
    assert (indexed['heat_events'],indexed['radiation_events'])==(5,4)
    cases['step-output']=indexed

    for name,end in [('heat-tail',.025),('dom-tail',.03)]:
        tail=simulate(args,name,args.binary,adaptive=False,dt=.0005,end=end,period=.01,write=1,
                      frequency=2,log_interval=5000)
        assert (tail['heat_events'],tail['radiation_events'])==(3,2)
        assert (tail['snapshots'][end]['heat'],tail['snapshots'][end]['radiation'])==(end,end)
        cases[name]=tail

    host=simulate(args,'host-clock',args.binary,adaptive=False,dt=5e-6,end=2.5e-5,period=1e-5,
                  write=600,write_control='clockTime',log_interval=5000)
    assert (host['heat_events'],host['radiation_events'])==(3,2) and set(host['snapshots'])=={2.5e-5}
    cases['host-clock']=host

    archived=args.output/'legacy-restart'/'0.29500248649130967'
    shutil.copytree(args.output/'aligned'/aligned['snapshots'][.005]['directory'],archived)
    t=float(archived.name)
    manifest=archived/'thermalExchangeState'
    for key,value in dict(exchangeSequence='"59"',completedSimulationTimeS=t,
                          completedRadiationSimulationTimeS=.290002,previousExchangeSimulationTimeS=.290002).items():
        replace_entry(manifest,key,value)
    for key in ('checkpointSimulationTimeS','completedExchangeSimulationTimeS'):
        replace_entry(archived/'gpuPendingWallEnergyLedgers.dat',key,t)
    replace_entry(archived/'uniform/time','name','"'+archived.name+'"')
    replace_entry(archived/'uniform/time','value',t)
    resumed=simulate(args,'to-0.3',args.binary,adaptive=False,dt=7e-5,end=.3,period=.005,write=-1,restart=archived)
    assert resumed['final']==.3 and set(resumed['snapshots'])=={.3}
    end=resumed['snapshots'][.3]
    assert (end['sequence'],end['heat'],end['radiation'],end['last_exchange'])==(60,.3,.3,t)
    assert (resumed['heat_events'],resumed['radiation_events'])==(1,1)
    cases['late-restart']=resumed
    (args.output/'results.json').write_text(json.dumps(cases,indent=2)+'\n')
    for name,c in cases.items():
        print(name,c['steps'],c['first_dt'],c['final'],sorted(c['snapshots']))


if __name__=='__main__':
    main()
