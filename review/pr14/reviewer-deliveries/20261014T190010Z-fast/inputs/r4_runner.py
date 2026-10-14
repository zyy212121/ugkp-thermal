from pathlib import Path
import subprocess,json,hashlib,datetime,time,os,shlex,fcntl,shutil
P=Path(__file__).resolve().parent;PATHS=json.loads((P/'r4_paths.json').read_text())
ROOT=Path(PATHS['checkout']);D=Path(PATHS['delivery']);OLDROOT=Path(PATHS['old_checkout']);OLD=Path(PATHS['old_delivery']);DEV=ROOT/PATHS['developer_directory'];SOURCE=PATHS['source']
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
 names={str(p.relative_to(OLD)) for p in (OLD/prefix).rglob('*') if p.is_file()}
 names|={q['alias'] for q in json.loads((OLD/'metadata/immutable-references.json').read_text())['references'] if q['alias'].startswith(prefix.rstrip('/')+'/')}
 rows=[]
 for n in sorted(names):
  p=Path(dest)/Path(n).relative_to(prefix);p.parent.mkdir(parents=True,exist_ok=True);b=old_read(n);p.write_bytes(b);rows.append({'path':str(p),'old_alias':n,'sha256':hashlib.sha256(b).hexdigest()})
 assert rows,prefix
 return rows
def amend(testid,**kw):
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);data=json.loads((D/'results.json').read_text());next(q for q in data['tests'] if q['id']==testid).update(kw);(D/'results.json').write_text(json.dumps(data,indent=2)+'\n')
def resource_snapshot(pid,log):
 snapshot={'utc':utc(),'log_bytes':log.stat().st_size,'processes':[]}
 ps=subprocess.run(['ps','-eo','pid,ppid,rss,etimes,comm'],capture_output=True,text=True).stdout.splitlines()[1:]
 rows=[]
 for line in ps:
  a=line.split(None,4)
  if len(a)==5:rows.append({'pid':int(a[0]),'ppid':int(a[1]),'rss_kib':int(a[2]),'elapsed_seconds':int(a[3]),'comm':a[4]})
 descendants={pid}
 while True:
  extended=descendants|{q['pid'] for q in rows if q['ppid'] in descendants}
  if extended==descendants:break
  descendants=extended
 snapshot['processes']=[q for q in rows if q['pid'] in descendants]
 snapshot['aggregate_rss_kib']=sum(q['rss_kib'] for q in snapshot['processes'])
 snapshot['meminfo']={a:b.strip() for a,b in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if a in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}
 snapshot['vmstat_oom_kill']=next((line.split()[1] for line in Path('/proc/vmstat').read_text().splitlines() if line.startswith('oom_kill ')),None)
 snapshot['gpu']=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
 for p in [Path('/sys/fs/cgroup/memory.events'),Path('/sys/fs/cgroup/memory.current'),Path('/sys/fs/cgroup/memory.peak')]:
  if p.is_file():snapshot[str(p)]=p.read_text().strip()
 return snapshot
def run(testid,args,topics=(),expected='',inputs=(),binary=None,variant='FORMAL',kind='GPU',cwd=None,source=None,progress_dirs=(),estimate_seconds=None):
 log=D/'logs'/f'{testid}.log';assert not log.exists(),testid+' already recorded';source=source or SOURCE
 targets=[];idx=json.loads((ROOT/'review/pr14/source-index.json').read_text())
 for f in idx['files']:
  if any(f'R014-{t:03d}' in f['topic_ids'] for t in topics):
   p=ROOT/f['path']
   if not p.is_file():continue
   blob=git('rev-parse',source+':'+f['path'])
   targets.append({'file_id':f['id'],'path':f['path'],'blob':blob,'sha256':sha(p),'function_ids':[q['id'] for q in f.get('functions',[])]})
 times=D/'logs'/f'{testid}.resource-summary.txt'
 actual=['/usr/bin/time','-v','-o',str(times),*map(str,args)]
 spec={'id':testid,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','kind':kind,'variant':variant,'source_commit':source,'source_tree':git('rev-parse',source+'^{tree}'),'source_hash_manifest':'metadata/source-hashes.json','source_hash_manifest_sha256':sha(D/'metadata/source-hashes.json'),'source_targets':targets,'topics':[f'R014-{t:03d}' for t in topics],'argv':actual,'working_directory':str(cwd or ROOT),'automatic_time_limit':None,'estimate_seconds_for_scheduling_only':estimate_seconds,'expected':expected,'input_hashes':{str(p):sha(p) for p in map(Path,inputs) if p.is_file()},'binary':str(binary) if binary else None,'binary_sha256':sha(binary) if binary and Path(binary).is_file() else None,'environment_script_sha256':sha(D/'inputs/env.sh'),'enqueued_utc':utc(),'progress_monitor':'External30s process/RAM/GPU/log metadata only; no automatic termination or solver changes.'}
 lock=(ROOT.parent/'gpu-fast.lock').open('a')
 if kind in ['GPU','SANITIZER','DIAGNOSTIC','RESOURCE_GPU']:fcntl.flock(lock,fcntl.LOCK_EX)
 spec['started_utc']=utc()
 (D/'logs'/f'{testid}.spec.json').write_text(json.dumps(spec,indent=2)+'\n')
 with (D/'commands.txt').open('a') as h:h.write('\nEXECUTED '+testid+'\nUTC '+spec['started_utc']+'\ncwd '+spec['working_directory']+'\n'+shlex.join(actual)+'\nautomatic_time_limit NONE\n')
 print('START '+testid,flush=True);start=time.monotonic();observations=[]
 with log.open('wb') as h:
  proc=subprocess.Popen(actual,cwd=spec['working_directory'],stdout=h,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1'))
  observations.append(resource_snapshot(proc.pid,log))
  while True:
   try:code=proc.wait(timeout=30);break
   except subprocess.TimeoutExpired:
    s=resource_snapshot(proc.pid,log);s['elapsed_seconds']=time.monotonic()-start;s['progress']={}
    for folder in map(Path,progress_dirs):s['progress'][str(folder)]=[{'path':str(p.relative_to(folder)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in folder.rglob('*') if p.is_file() and p.name in ['accepted-timeline.csv','timeline.csv','state.chmt','accepted.csv']]
    observations.append(s);(D/'logs'/f'{testid}.progress.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('PROGRESS '+testid+' seconds='+str(round(s['elapsed_seconds'],1))+' logbytes='+str(s['log_bytes'])+' rssKiB='+str(s['aggregate_rss_kib'])+' '+s['gpu'],flush=True)
  observations.append(resource_snapshot(proc.pid,log))
 (D/'logs'/f'{testid}.progress.json').write_text(json.dumps(observations,indent=2)+'\n');lock.close()
 result={**spec,'ended_utc':utc(),'elapsed_seconds':time.monotonic()-start,'exit_code':code,'signal':-code if code<0 else None,'external_timeout':False,'log':str(log.relative_to(D)),'log_sha256':sha(log),'resource_summary':str(times.relative_to(D)),'verdict':'PASS' if kind=='BUILD' and code==0 else 'INCONCLUSIVE' if code==0 else 'FAIL','observations':'Actual process observation; numeric acceptance requires separate full assertions.','artifacts':[]}
 with (ROOT/'work/results.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);data=json.loads((D/'results.json').read_text());data['tests'].append(result);(D/'results.json').write_text(json.dumps(data,indent=2)+'\n')
 print('END '+testid+' exit='+str(code)+' elapsed='+str(round(result['elapsed_seconds'],3)),flush=True)
 if code:print(log.read_text(errors='replace')[-3500:],flush=True)
 return result
def assertion(t,ok,observation,**kw):amend(t['id'],verdict='PASS' if ok and t['exit_code']==0 else 'FAIL',observations=observation,**kw)
def archive_output(case,name):
 out=case/'chmtOutput';dest=D/'artifacts'/name
 if out.is_dir():shutil.copytree(out,dest,dirs_exist_ok=True)
 return dest
