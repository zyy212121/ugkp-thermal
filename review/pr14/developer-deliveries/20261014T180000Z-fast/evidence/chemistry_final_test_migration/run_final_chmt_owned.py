from pathlib import Path
import hashlib,json,subprocess,time,datetime,resource
r=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');o=r.parent/'pr14_core_chemistry_test_adaptation_20261014'
files=['applications/CHMT/tests/test_'+x+'.py' for x in ['final_gas_consumers','complete_payload_admission','coupled_runtime_transactions','shared_reconfiguration']]
cmd=['/tmp/pr14_validation_venv/bin/pytest','-q',*files]
paths=subprocess.check_output(['git','ls-files','common','applications/CHMT'],cwd=r,text=True).splitlines()
def snapshot():return {x:hashlib.sha256((r/x).read_bytes()).hexdigest() for x in paths if (r/x).is_file() and Path(x).suffix in {'.py','.H','.C','.h','.cpp','.cuh','.cu','.in'}}
before=snapshot();(o/'chmt-frozen-source-before.json').write_text(json.dumps(before,indent=2)+'\n');started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with (o/'chmt-frozen.log').open('w') as f:rc=subprocess.run(cmd,cwd=r,stdout=f,stderr=subprocess.STDOUT).returncode
after=snapshot();(o/'chmt-frozen-source-after.json').write_text(json.dumps(after,indent=2)+'\n')
receipt={'command':cmd,'cwd':str(r),'started_at_utc':started,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':rc,'elapsed_seconds':time.monotonic()-t,'maximum_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'source_count':len(before),'sources_unchanged':before==after,'changed_sources':[x for x in before if before[x]!=after.get(x)],'gpu_execution':False,'validation_kind':'host CUDA allocation shims and serial translated production kernels; not native GPU evidence'}
(o/'chmt-frozen-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));print((o/'chmt-frozen.log').read_text()[-5000:])
