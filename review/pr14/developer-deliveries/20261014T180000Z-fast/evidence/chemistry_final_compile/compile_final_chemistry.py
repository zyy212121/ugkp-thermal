import pathlib,subprocess,json,hashlib,time,datetime,resource,sys
r=pathlib.Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');o=r.parent/'pr14_fast4_chemistry_final_compile_20261014';name=sys.argv[1]
cmd=json.loads((o/'command.json').read_text())
if name=='device-low-opt':
 cmd=[x if x!='-O3' else '-O0' for x in cmd];cmd[1:1]=['--Ofast-compile=max','-Xptxas=-O0'];cmd[-1]+='-device-low-opt'
paths=subprocess.check_output(['git','ls-files','common','applications/gasUGKP/private_backend','applications/gasUGKP/gpu','tests/fixtures/chemistry_trial','tests/test_chemistry_trial_cuda_fixture.py'],cwd=r,text=True).splitlines()
def snap():return {x:hashlib.sha256((r/x).read_bytes()).hexdigest() if (r/x).is_file() else None for x in paths if pathlib.Path(x).suffix in {'.C','.H','.cu','.cuh','.h','.cpp','.inc','.inl','.py','.in'}}
before=snap();(o/(name+'-source-before.json')).write_text(json.dumps(before,indent=2)+'\n');(o/(name+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (o/(name+'-build.log')).open('w') as f:rc=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT).returncode
after=snap();(o/(name+'-source-after.json')).write_text(json.dumps(after,indent=2)+'\n')
rct={'name':name,'command':cmd,'cwd':str(r),'started_at_utc':start,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'exit_code':rc,'maximum_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'source_before_after_equal':before==after,'source_changes_during_run':[x for x in paths if before.get(x)!=after.get(x)],'source_count':len(before),'generated_source_sha256':hashlib.sha256((o/'chemistry_trial.cu').read_bytes()).hexdigest(),'gpu_runtime_executed':False,'original_O3_acceptance':name=='O3' and rc==0,'full_production_fixture_compiled':rc==0}
if rc==0:rct['binary_sha256']=hashlib.sha256(pathlib.Path(cmd[-1]).read_bytes()).hexdigest()
(o/(name+'-receipt.json')).write_text(json.dumps(rct,indent=2)+'\n');print(json.dumps(rct))
