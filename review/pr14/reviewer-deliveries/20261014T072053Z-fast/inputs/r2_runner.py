import pathlib,subprocess,json,hashlib,datetime,time,os,signal,shlex,fcntl,shutil
P=pathlib.Path(__file__).resolve().parent
PATHS=json.loads((P/'r2_paths.json' if (P/'r2_paths.json').exists() else P.parent/'SOURCE.json').read_text())
ROOT=pathlib.Path(PATHS['checkout']);D=pathlib.Path(PATHS['delivery']);OLD=pathlib.Path(PATHS['old_delivery']);OLDROOT=pathlib.Path(PATHS['old_checkout']);DEV=ROOT/PATHS['developer_directory'];SOURCE=PATHS['source']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
def cmd(*a):return ['bash',str(D/'inputs/env.sh'),*map(str,a)]
def run(testid,args,topics=(),budget=300,expected='',inputs=(),binary=None,variant='formal',kind='GPU',cwd=None,source=None):
 log=D/'logs'/f'{testid}.log';assert not log.exists(),testid+' already recorded'
 source=source or (PATHS['old_source'] if variant=='old-production' else PATHS['baseline'] if variant=='baseline' else SOURCE)
 src=json.loads((ROOT/'review/pr14/source-index.json').read_text())
 targets=[]
 for f in src['files']:
  if any(f'R014-{t:03d}' in f['topic_ids'] for t in topics):
   p=ROOT/f['path'];targets.append({'file_id':f['id'],'path':f['path'],'sha256':sha(p) if p.is_file() else None,'function_ids':[x['id'] for x in f.get('functions',[])]})
 hashes={str(p):sha(p) for p in map(pathlib.Path,inputs) if p.is_file()}
 spec={'id':testid,'interaction_class':'FAST','kind':kind,'variant':variant,'source_commit':source,'source_tree':git('rev-parse',source+'^{tree}'),'source_hash_manifest':'metadata/source-hashes.json','source_hash_manifest_sha256':sha(D/'metadata/source-hashes.json'),'diagnostic_patch':variant if 'diagnostic' in variant else None,'topics':[f'R014-{t:03d}' for t in topics],'source_targets':targets,'argv':list(map(str,args)),'working_directory':str(cwd or ROOT),'budget_seconds':budget,'expected':expected,'input_hashes':hashes,'binary':str(binary) if binary else None,'binary_sha256':sha(binary) if binary and pathlib.Path(binary).is_file() else None,'environment_script_sha256':sha(D/'inputs/env.sh'),'started_utc':utc()}
 (D/'logs'/f'{testid}.spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 with (D/'commands.txt').open('a') as f:f.write('\nEXECUTED '+testid+'\nUTC '+spec['started_utc']+'\ncwd '+spec['working_directory']+'\n'+shlex.join(spec['argv'])+'\nexternal_budget_seconds '+str(budget)+'\n')
 print('START '+testid,flush=True);t=time.monotonic();timeout=False
 with log.open('wb') as f:
  proc=subprocess.Popen(spec['argv'],cwd=spec['working_directory'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1'))
  try:code=proc.wait(timeout=budget)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=124
 result={**spec,'ended_utc':utc(),'elapsed_seconds':time.monotonic()-t,'exit_code':code,'external_timeout':timeout,'log':str(log.relative_to(D)),'log_sha256':sha(log),'verdict':'INCONCLUSIVE' if code==0 else 'BLOCKED' if timeout else 'FAIL','observations':'Captured process status; scoped acceptance requires independent fields/assertions.','artifacts':[]}
 if kind=='BUILD' and code==0:result.update(verdict='PASS',observations='Native compilation/link complete, runtime acceptance separate.')
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);a=json.loads((D/'results.json').read_text());a['tests'].append(result);(D/'results.json').write_text(json.dumps(a,indent=2)+'\n')
 print('END '+testid+' exit='+str(code)+' seconds='+str(round(result['elapsed_seconds'],3)),flush=True)
 if code:print(log.read_text(errors='replace')[-6000:],flush=True)
 return result
def amend(testid,**kw):
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);a=json.loads((D/'results.json').read_text());next(x for x in a['tests'] if x['id']==testid).update(kw);(D/'results.json').write_text(json.dumps(a,indent=2)+'\n')
def assertion(t,ok,observation,**kw):amend(t['id'],verdict='PASS' if ok and t['exit_code']==0 else 'FAIL',observations=observation,**kw)
def save_case(case,name,kind='resident'):
 dest=D/'artifacts'/kind/name;dest.mkdir(parents=True,exist_ok=True)
 if kind=='chmt':shutil.copytree(case/'chmtOutput',dest,dirs_exist_ok=True)
 else:
  for q in case.iterdir():
   try:v=float(q.name)
   except ValueError:continue
   if v>0 and q.is_dir():shutil.copytree(q,dest/q.name)
 return dest
if __name__=='__main__':
 for app,script in [('CHMT2','applications/CHMT/Allwmake'),('CHT','applications/CHT/tools/build_cuda_solver.sh'),('FSH','applications/FSH/private_backend/build_private_backend.sh')]:
  run('BUILD-'+app,cmd('bash',ROOT/script),[10,12,23,24,25,26,33],900,'Fresh original native compile/link from fixed formal source; isolated installation.',inputs=[ROOT/script],kind='BUILD')
 run('BUILD-CHMT10',cmd('env','UGKWP_GAS_SPECIES=10',f'CHMT_BUILD={ROOT}/work/build/CHMT10',f'FOAM_USER_APPBIN={ROOT}/work/install/ns10/bin','bash',ROOT/'applications/CHMT/Allwmake'),[10,24,25,26],900,'Affected Ns10 native compile/link.',inputs=[ROOT/'applications/CHMT/Allwmake'],kind='BUILD')
