#!/usr/bin/env python3
from pathlib import Path
import datetime,fcntl,gzip,json,os,queue,re,subprocess,threading,time

case=Path(__file__).resolve().parent
logs=case.parent/'log'/case.name
logs.mkdir(parents=True,exist_ok=True)
lock=(logs/'run.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
tag=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
state={'case':str(case),'started':tag,'mode':'fully_transient','ranks':4,'status':'starting'}
def save_state():
    p=logs/'current.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'current.json')
save_state()
pending=queue.Queue()
def reconstruct():
    with (logs/('reconstruct_'+tag+'.log')).open('w',buffering=1) as out:
        while True:
            name=pending.get()
            if name is None:
                pending.task_done()
                return
            result=subprocess.run(['reconstructPar','-case',str(case),'-allRegions','-time',name,'-noFunctionObjects'],stdout=out,stderr=subprocess.STDOUT)
            receipt={'time':name,'exit':result.returncode,'recorded':datetime.datetime.now().isoformat()}
            with (logs/'written_times.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
            if result.returncode==0 and abs(float(name)-1.2)<1e-7:
                with (logs/'draw_at_1p2.log').open('w') as f:
                    subprocess.run(['python3',str(case.parent/'MSS7_turbulent_wallModel/draw.py')],stdout=f,stderr=subprocess.STDOUT)
                    subprocess.run(['python3',str(case/'draw.py'),'--only','wall-heat-flux'],stdout=f,stderr=subprocess.STDOUT)
                (logs/'comparison_1p2_ready.json').write_text(json.dumps({'time':1.2,'status':'data_ready_for_review_not_a_physics_pass'},indent=2))
            pending.task_done()
worker=threading.Thread(target=reconstruct)
worker.start()
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=['mpirun','-np','4','chtMultiRegionFoam','-parallel','-case',str(case)]
state['command']=command
process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
state.update(status='running',pid=process.pid)
save_state()
steps=0;name=None;selected=True;scheduled=set()
try:
    with gzip.open(logs/('CHT_'+tag+'.full.log.gz'),'wt',compresslevel=1) as full,(logs/'CHT.log').open('a',buffering=1) as compact:
        compact.write('\nSTART '+tag+' '+str(command)+'\n')
        for line in process.stdout:
            full.write(line)
            m=re.match(r'^Time = ([+\-0-9.eE]+)s?\s*$',line.strip())
            if m:
                name=m[1];steps+=1
                selected=steps<=3 or steps%1000==0
            if selected or any(word in line for word in ('FATAL','Warning','FOAM exiting','FOAM aborting','y+ :','End','Finalising')):
                compact.write(line)
            if line.startswith('ExecutionTime') and name is not None:
                if selected:
                    state.update(simulation_time=float(name),steps_this_run=steps,updated=datetime.datetime.now().isoformat())
                    save_state()
                if name not in scheduled and (case/'processor0'/name/'fluid/T').is_file():
                    scheduled.add(name);pending.put(name)
                    compact.write('WRITE_COMPLETE time='+name+' reconstruction queued\n')
        code=process.wait()
        compact.write('EXIT '+str(code)+'\n')
    pending.join()
    state.update(status='completed' if code==0 else 'failed',exit=code,simulation_time=float(name) if name else None,finished=datetime.datetime.now().isoformat())
    save_state()
finally:
    pending.put(None)
    worker.join()
