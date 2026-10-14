import json,collections,subprocess,shutil,re,hashlib,os
from r2_runner import *
print('FILES',[(p.name,p.stat().st_size) for p in (D/'metadata').iterdir() if p.is_file()])
old=json.loads((OLD/'results.json').read_text());print('OLD UNITS',[(q.get('id'),q.get('scope','')[:130],q.get('verdict'),q.get('topics')) for q in old.get('acceptance_units',[])][:140])
print('OLD result keys',list(old))
print('RESOURCE HEAD',{k:v for k,v in json.loads((D/'artifacts/nsys/resource-analysis.json').read_text()).items() if k not in ['Co_reduction_NVTX_ranges','all_mem_events','memory_events','all_memory_events','Co_warm_first_pair'] and not isinstance(v,list)})
# Runtime and allfreshproducing binaries/objects (large compiledpayloads remainlocal).
tests=json.loads((D/'results.json').read_text())['tests'];binaries={}
for t in tests:
 if t.get('binary') and pathlib.Path(t['binary']).is_file():binaries[t['binary']]={'sha256':sha(t['binary']),'bytes':pathlib.Path(t['binary']).stat().st_size,'test_ids':[q['id'] for q in tests if q.get('binary')==t['binary']]}
objects={str(p.relative_to(ROOT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [ROOT/'work/build/CHMT2/Backend.o',ROOT/'work/build/CHMT10/Backend.o',ROOT/'work/fixtures/Backend-ledger2.o',ROOT/'applications/CHT/gpu/libugkpcht_cuda_internal.a',ROOT/'applications/FSH/private_backend/libugkwp_cuda_backend.a'] if p.is_file()}
(D/'versions.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'source_tree':PATHS['source_tree'],'build_flags':'Exactpertestargv/spec: native Ns2/Ns10,sm89, physicalFP64; CHTtwo coldfloatprofiles +doublecarry;FSHdouble. Originalsuccess fmad flags perapplicationcommands; no precision change.','binaries':binaries,'objects':objects,'diagnostic_patches':{str(p.relative_to(D)):sha(p) for p in (D/'patches').glob('*.patch')},'scope':'Hashes of actualused source/object/binary; compilationPASS neverGPU numericalacceptance; compiledpayloads notuploaded.'},indent=2)+'\n')
env={}
for name,args in [('gpu',['/usr/lib/wsl/lib/nvidia-smi']),('nvcc',['/usr/local/cuda/bin/nvcc','--version']),('host',['g++','--version']),('uname',['uname','-a']),('nsys',['nsys','--version']),('sanitizer',['compute-sanitizer','--version'])]:
 p=subprocess.run(cmd(*args),capture_output=True,text=True);env[name]={'exit':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
env.update(identity='PR14_REVIEW_ONLY',captured_utc=utc(),execution='Ubuntu22.04 WSL2 lss; actualsingleRTX4060Laptop8GB; isolated test worktree/install; no productionlongrun',crash_event='metadata/crash-resume.json',environment_script='inputs/env.sh',environment_sha256=sha(D/'inputs/env.sh'))
(D/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
# Archive exacthelpers; configurablefilesystemroots aretheuserauthorizedtestarea.
for p in P.glob('r2_*'):
 if p.is_file() and p.name!='r2_paths.json':shutil.copy2(p,D/'inputs'/p.name)
shutil.copy2(P/'r2_paths.json',D/'inputs/r2_paths.json')
snapshot=[{'path':str(p.relative_to(D)),'sha256':sha(p),'bytes':p.stat().st_size} for p in (D/'inputs').glob('r2_*') if p.is_file()]
(D/'metadata/driver-scripts.json').write_text(json.dumps(snapshot,indent=2)+'\n')
# Formal/source andlog integrities, before admin edits.
bad=[t['id'] for t in tests if sha(D/t['log'])!=t['log_sha256']];assert not bad
src=json.loads((D/'metadata/source-hashes.json').read_text());assert all(sha(ROOT/x['path'])==x['sha256'] for x in src);assert not git('diff','--stat','--','applications','common','tests','tools')
(D/'metadata/final-source-and-log-integrity.json').write_text(json.dumps({'verified_utc':utc(),'production_source_commit':SOURCE,'source_file_count':len(src),'all_source_hashes_match':True,'all_logs_hash_match':True,'record_count':len(tests),'command_verdicts':dict(collections.Counter(t['verdict'] for t in tests)),'production_tracked_diff_empty':True,'note':'Rawcommand counts include BUILD/setup/analysis/supportfailures; notGPU numericacceptancecounts.'},indent=2)+'\n')
print('PREPARED',len(tests),'sources',len(src),'binaries',len(binaries),flush=True)
