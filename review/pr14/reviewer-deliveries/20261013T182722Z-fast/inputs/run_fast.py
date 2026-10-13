#!/usr/bin/env python3
import pathlib,subprocess,json,hashlib,datetime,time,os,signal,sys,shlex,fcntl
if (pathlib.Path(__file__).parent/'paths.json').exists():
    PATHS=json.loads((pathlib.Path(__file__).parent/'paths.json').read_text());ROOT=pathlib.Path(PATHS['checkout']);D=pathlib.Path(PATHS['delivery'])
else:
    D=pathlib.Path(__file__).resolve().parent.parent
    ROOT=D.parents[3]
ORIGINAL='802a11e7dfe986488a8bf8305e86424dcc94c9e1'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def run(testid,args,topics,budget=300,expected='',inputs=(),binary=None,variant='original',kind='GPU',cwd=None):
    cwd=str(cwd or ROOT);log=D/'logs'/f'{testid}.log'
    assert not log.exists(), testid+' evidence already exists'
    src=json.loads((ROOT/'review/pr14/source-index.json').read_text())
    targets=[{'file_id':f['id'],'path':f['path'],'blob':f.get('head_blob'),'sha256':f.get('worktree_sha256'),'function_ids':[a.get('id') for a in f.get('functions',[])]} for f in src['files'] if any(f'R014-{t:03d}' in f['topic_ids'] for t in topics)]
    ih={str(pathlib.Path(p).relative_to(ROOT)) if pathlib.Path(p).is_relative_to(ROOT) else str(p):sha(p) for p in inputs if pathlib.Path(p).is_file()}
    spec={'id':testid,'interaction_class':'FAST','kind':kind,'variant':variant,'source_commit':ORIGINAL if variant=='original' else git('rev-parse','HEAD'),'source_tree':'db2863158139842f6c8221c0da11ba640f381336','baseline_commit':'d6110c02745a8ad726208a426c4bb3a28233c1fd','topics':[f'R014-{t:03d}' for t in topics],'source_targets':targets,'argv':[str(a) for a in args],'working_directory':cwd,'budget_seconds':budget,'expected':expected,'input_hashes':ih,'binary':str(binary) if binary else None,'binary_sha256':sha(binary) if binary and pathlib.Path(binary).is_file() else None,'environment_script_sha256':sha(D/'inputs/env.sh'),'started_utc':utc()}
    (D/'logs'/f'{testid}.spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
    with (D/'commands.txt').open('a') as f:f.write(f'\nEXECUTED {testid}\nUTC {spec["started_utc"]}\ncwd {cwd}\n{shlex.join(spec["argv"])}\nexternal_budget_seconds {budget}\n')
    print('START '+testid,flush=True);t=time.monotonic();timedout=False
    with log.open('wb') as f:
        p=subprocess.Popen(spec['argv'],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1'))
        try:code=p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timedout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=124
    result={**spec,'ended_utc':utc(),'elapsed_seconds':time.monotonic()-t,'exit_code':code,'external_timeout':timedout,'log':str(log.relative_to(D)),'log_sha256':sha(log),'verdict':'INCONCLUSIVE' if code==0 else 'BLOCKED' if timedout else 'FAIL','observations':'Exit code captured; acceptance requires reading fields/checker results.','artifacts':[]}
    if kind=='BUILD' and code==0: result['verdict']='PASS';result['observations']='Native build script child returned exit 0. Runtime and numerical acceptance are separate.'
    with (ROOT/'work/results.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        results=json.loads((D/'results.json').read_text());results['tests'].append(result);(D/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(f'END {testid} exit={code} seconds={result["elapsed_seconds"]:.3f} log={log}',flush=True)
    if code: print(log.read_text(errors='replace')[-7000:],flush=True)
    return result
def amend(testid,**fields):
    with (ROOT/'work/results.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads((D/'results.json').read_text());t=next(t for t in data['tests'] if t['id']==testid);t.update(fields);(D/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def cmd(*args): return ['bash',str(D/'inputs/env.sh'),*map(str,args)]
def builds(suffix=''):
    for app,script in [('CHMT2','applications/CHMT/Allwmake'),('gasUGKP2','applications/gasUGKP/private_backend/build_private_backend.sh'),('CHT','applications/CHT/tools/build_cuda_solver.sh'),('FSH','applications/FSH/private_backend/build_private_backend.sh')]:
        run('BUILD-'+app+suffix,cmd('bash',str(ROOT/script)),[24,25],900,'Original native build entry returns 0; separate isolated installed binary, full linker/CUDA compile log.',inputs=[ROOT/script],kind='BUILD')
    run('BUILD-CHMT10'+suffix,cmd('env','UGKWP_GAS_SPECIES=10',f'CHMT_BUILD={ROOT}/work/build/CHMT10',f'FOAM_USER_APPBIN={ROOT}/work/install/ns10/bin','bash',str(ROOT/'applications/CHMT/Allwmake')),[10,24,25],900,'Ns10 native complete build/link exit0, isolated species10 binary.',inputs=[ROOT/'applications/CHMT/Allwmake'],kind='BUILD')
if __name__=='__main__' and sys.argv[1]=='build': builds(sys.argv[2] if len(sys.argv)>2 else '')
