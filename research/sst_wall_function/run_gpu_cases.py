"""Run unchanged retained gasUGKP/CHMT GPU cases and verify terminal output.

Preparation is NOT_RUN. Actual solver execution needs --execute and explicit
binary paths; no shortened endpoint, runtime timeout, or stopped-run success.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,re,shutil,subprocess,time
HERE=Path(__file__).parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--case',choices=['gas_flatplate','chmt_mss7_moving'],required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--solver',type=Path)
 p.add_argument('--backend',type=Path,help='Required gasUGKPCudaBackend identity for gas_flatplate')
 p.add_argument('--execute',action='store_true');a=p.parse_args()
 src=HERE/'gpu_inputs'/a.case;manifest=json.loads((src/'input_manifest.json').read_text())
 for name,expected in manifest['files'].items():
  if digest(src/name)!=expected:raise RuntimeError('retained input changed: '+name)
 a.output.mkdir(parents=True,exist_ok=False);case=a.output/'case';shutil.copytree(src,case)
 control=(case/'system/controlDict').read_text();endpoint=float(re.search(r'\bendTime\s+([^;]+);',control).group(1))
 expected=.2 if a.case=='gas_flatplate' else .01
 if endpoint!=expected:raise RuntimeError('original endpoint changed')
 record={'case':a.case,'endpoint':endpoint,'inputs_sha256':digest(src/'input_manifest.json'),'status':'NOT_RUN','scope':'full retained GPU case; no host reference substitution'}
 result=a.output/'result.json'
 if not a.execute:
  result.write_text(json.dumps(record,indent=2));print(json.dumps(record));return
 if not a.solver or not a.solver.is_file():raise RuntimeError('explicit solver binary required')
 env=os.environ.copy();record['solver']=str(a.solver.resolve());record['solver_sha256']=digest(a.solver)
 if a.case=='gas_flatplate':
  if not a.backend or not a.backend.is_file():raise RuntimeError('explicit gasUGKPCudaBackend binary required')
  env['GAS_UGKP_CUDA_BACKEND']=str(a.backend.resolve());record['backend']=str(a.backend.resolve());record['backend_sha256']=digest(a.backend)
 command=[str(a.solver.resolve()),'-case',str(case.resolve())];record['command']=command
 record['status']='RUNNING';result.write_text(json.dumps(record,indent=2));started=time.monotonic()
 with (a.output/'solver.log').open('w') as log:run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 record.update(returncode=run.returncode,wall_seconds=time.monotonic()-started,status='FAIL')
 if run.returncode==0:
  if a.case=='gas_flatplate':
   times=[(float(q.name),q) for q in case.iterdir() if q.is_dir() and re.fullmatch(r'[0-9.eE+-]+',q.name)]
   final,folder=max(times);record['actual_final_time']=final
   required=['rho','p','T','U','rhoE','Y_A','Y_B','k','omega']
   complete=abs(final-endpoint)<=1e-12 and all((folder/name).is_file() for name in required)
  else:
   states=[]
   for file in case.glob('chmtOutput/accepted-*.csv'):
    states+=list(csv.DictReader(file.open()))
   final=max(states,key=lambda x:float(x['time'])) if states else None
   record['final_accepted']=final
   complete=bool(final) and abs(float(final['time'])-endpoint)<=1e-12 and int(final['accepted_steps'])>0
  if complete:record['status']='PASS_ENDPOINT_ONLY'
 if a.backend:
  record['backend_sha256_after']=digest(a.backend)
  if record['backend_sha256_after']!=record.get('backend_sha256'):record['status']='FAIL_BINARY_CHANGED'
 record['solver_sha256_after']=digest(a.solver)
 if record['solver_sha256_after']!=record['solver_sha256']:record['status']='FAIL_BINARY_CHANGED'
 result.write_text(json.dumps(record,indent=2));print(json.dumps(record))
 if record['status']!='PASS_ENDPOINT_ONLY':raise SystemExit(1)
if __name__=='__main__':main()
