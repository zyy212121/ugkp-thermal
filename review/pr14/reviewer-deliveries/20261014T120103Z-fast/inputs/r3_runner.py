from pathlib import Path
import subprocess,json,hashlib,datetime,time,os,signal,shlex,fcntl,shutil
P=Path(__file__).resolve().parent;PATHS=json.loads((P/'r3_paths.json').read_text());ROOT=Path(PATHS['checkout']);D=Path(PATHS['delivery']);OLDROOT=Path(PATHS['old_checkout']);OLD=Path(PATHS['old_delivery']);DEV=ROOT/PATHS['developer_directory'];SOURCE=PATHS['source']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
def cmd(*a):return ['bash',str(D/'inputs/env.sh'),*map(str,a)]
def old_read(name):
 p=OLD/name
 if p.is_file():return p.read_bytes()
 ref=next(q for q in json.loads((OLD/'metadata/immutable-references.json').read_text())['references'] if q['alias']==name)
 payload=subprocess.check_output(['git','-C',str(ROOT),'show',ref['fixed_commit']+':'+ref['repo_path']]);assert hashlib.sha256(payload).hexdigest()==ref['sha256'];return payload
def restore_old_prefix(prefix,dest):
 paths={str(p.relative_to(OLD)) for p in (OLD/prefix).rglob('*') if p.is_file()}
 paths|={q['alias'] for q in json.loads((OLD/'metadata/immutable-references.json').read_text())['references'] if q['alias'].startswith(prefix.rstrip('/')+'/')}
 records=[]
 for name in sorted(paths):
  p=Path(dest)/Path(name).relative_to(prefix);p.parent.mkdir(parents=True,exist_ok=True);payload=old_read(name);p.write_bytes(payload);records.append({'path':str(p),'old_alias':name,'sha256':hashlib.sha256(payload).hexdigest()})
 assert records,prefix
 return records
def amend(testid,**kw):
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);data=json.loads((D/'results.json').read_text());next(q for q in data['tests'] if q['id']==testid).update(kw);(D/'results.json').write_text(json.dumps(data,indent=2)+'\n')
def run(testid,args,topics=(),budget=300,expected='',inputs=(),binary=None,variant='FORMAL',kind='GPU',cwd=None,source=None,progress_dirs=()):
 log=D/'logs'/f'{testid}.log';assert not log.exists(),testid+' already recorded';source=source or SOURCE
 src=json.loads((ROOT/'review/pr14/source-index.json').read_text());targets=[]
 for f in src['files']:
  if any(f'R014-{t:03d}' in f['topic_ids'] for t in topics):
   try:blob=git('rev-parse',source+':'+f['path'])
   except subprocess.CalledProcessError:continue
   targets.append({'file_id':f['id'],'path':f['path'],'blob':blob,'sha256':hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'show',source+':'+f['path']])).hexdigest(),'function_ids':[q['id'] for q in f.get('functions',[])]})
 spec={'id':testid,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','kind':kind,'variant':variant,'source_commit':source,'source_tree':git('rev-parse',source+'^{tree}'),'source_hash_manifest':'metadata/source-hashes.json','source_hash_manifest_sha256':sha(D/'metadata/source-hashes.json'),'source_targets':targets,'topics':[f'R014-{t:03d}' for t in topics],'argv':list(map(str,args)),'working_directory':str(cwd or ROOT),'budget_seconds':budget,'expected':expected,'input_hashes':{str(p):sha(p) for p in map(Path,inputs) if p.is_file()},'binary':str(binary) if binary else None,'binary_sha256':sha(binary) if binary and Path(binary).is_file() else None,'environment_script_sha256':sha(D/'inputs/env.sh'),'started_utc':utc(),'progress_monitor':'External30s metadata snapshots only; no solver changes.'}
 (D/'logs'/f'{testid}.spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 with (D/'commands.txt').open('a') as h:h.write('\nEXECUTED '+testid+'\nUTC '+spec['started_utc']+'\ncwd '+spec['working_directory']+'\n'+shlex.join(spec['argv'])+'\nexternal_budget_seconds '+str(budget)+'\n')
 print('START '+testid,flush=True);start=time.monotonic();timeout=False;observations=[]
 lock=(ROOT.parent/'gpu-fast.lock').open('a')
 if kind in ['GPU','SANITIZER','DIAGNOSTIC','RESOURCE_GPU']:fcntl.flock(lock,fcntl.LOCK_EX)
 with log.open('wb') as h:
  proc=subprocess.Popen(spec['argv'],cwd=spec['working_directory'],stdout=h,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1'))
  while True:
   elapsed=time.monotonic()-start
   try:code=proc.wait(timeout=min(30,max(.001,budget-elapsed)));break
   except subprocess.TimeoutExpired:
    elapsed=time.monotonic()-start
    snapshot={'utc':utc(),'elapsed_seconds':elapsed,'log_bytes':log.stat().st_size,'paths':{}}
    for folder in map(Path,progress_dirs):snapshot['paths'][str(folder)]=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in folder.rglob('*') if p.is_file() and p.name in ['accepted-timeline.csv','timeline.csv','state.chmt','accepted.csv']]
    observations.append(snapshot);(D/'logs'/f'{testid}.progress.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('PROGRESS '+testid+' seconds='+str(round(elapsed,1))+' logbytes='+str(snapshot['log_bytes']),flush=True)
    if elapsed>=budget:
     timeout=True;os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
     code=124;break
 lock.close()
 result={**spec,'ended_utc':utc(),'elapsed_seconds':time.monotonic()-start,'exit_code':code,'signal':-code if code<0 else None,'external_timeout':timeout,'log':str(log.relative_to(D)),'log_sha256':sha(log),'verdict':'PASS' if kind=='BUILD' and code==0 else 'INCONCLUSIVE' if code==0 else 'BLOCKED' if timeout else 'FAIL','observations':'Actual process observation; numeric acceptance requires separate full assertions.','artifacts':[]}
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);data=json.loads((D/'results.json').read_text());data['tests'].append(result);(D/'results.json').write_text(json.dumps(data,indent=2)+'\n')
 print('END '+testid+' exit='+str(code)+' elapsed='+str(round(result['elapsed_seconds'],3)),flush=True)
 if code:print(log.read_text(errors='replace')[-3000:],flush=True)
 return result
def assertion(t,ok,observation,**kw):amend(t['id'],verdict='PASS' if ok and t['exit_code']==0 else 'BLOCKED' if t['external_timeout'] else 'FAIL',observations=observation,**kw)
def archive_output(case,name):
 out=case/'chmtOutput';target=D/'artifacts'/name
 if out.is_dir():shutil.copytree(out,target,dirs_exist_ok=True)
 return target
