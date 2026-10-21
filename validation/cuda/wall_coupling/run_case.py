#!/usr/bin/env python3
"""Run an isolated case with explicit private runtime and record reproducible timing."""
import argparse,hashlib,json,os,re,statistics,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--sanitize',choices=['memcheck','racecheck','synccheck']);ap.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-13.1'));a=ap.parse_args()
a.case=a.case.resolve();a.runtime=a.runtime.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,UGKP_WALL_AUDIT_DIR=str(a.output))
cmd=['bash',str(ROOT/'docs/diagnostics/bvp-coupling-20261020/run-case.sh'),str(a.runtime),str(a.case)]
if a.sanitize:cmd=[str(a.cuda/'bin/compute-sanitizer'),'--target-processes','all','--tool',a.sanitize,'--error-exitcode','99',*cmd]
def device():
 p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,pstate,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,memory.used,power.draw','--format=csv'],capture_output=True,text=True)
 return p.stdout+p.stderr
record={'command':cmd,'runtime_environment':{'GAS_UGKP_CUDA_STACK_BYTES':env.get('GAS_UGKP_CUDA_STACK_BYTES')},'device_before':device(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binaries':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.runtime/'bin').iterdir() if p.is_file()}}
t=time.monotonic();events=[]
with (a.output/'run.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,bufsize=1)
 for line in p.stdout:
  log.write(line);log.flush();m=re.search(r'UGKP_REPRO step=(\d+) time=([\deE.+-]+) dt=([\deE.+-]+)',line)
  if m:events.append({'step':int(m[1]),'time':float(m[2]),'dt':float(m[3]),'wall':time.monotonic()-t})
 rc=p.wait()
blocks=[(b['wall']-x['wall'])/(b['step']-x['step']) for x,b in zip(events[1:],events[2:])]
record.update(exit=rc,seconds=time.monotonic()-t,events=events,seconds_per_step=blocks,median_seconds_per_step=statistics.median(blocks) if blocks else None,device_after=device())
(a.output/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k in ['exit','seconds','median_seconds_per_step']}),flush=True)
raise SystemExit(rc)
