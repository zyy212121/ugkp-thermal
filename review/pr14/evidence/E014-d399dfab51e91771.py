from pathlib import Path
import json,hashlib,subprocess,difflib,shutil,datetime
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');v1=repo.parent/'chmt_remaining_audit_20261013/implementation/b5';out=v1/'v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
expected=['applications/CHMT/gpu/GasWallMath.H','applications/CHMT/tests/test_wall_terminal_packet_completion.py']
oldindex=json.loads((v1/'source-index.json').read_text());changed=[p for p,d in oldindex.items() if sha(repo/p)!=d['sha256']];assert changed==expected,changed
olddeps=json.loads((v1/'dependency-index.json').read_text());depschanged=[p for p,h in olddeps.items() if sha(repo/p)!=h];assert depschanged==[expected[0]],depschanged
oldartifacts=json.loads((v1/'evidence-index.json').read_text());assert len(oldartifacts)==4474
for p,d in oldartifacts.items():assert sha(v1/p)==d['sha256'],p
repair=''
for p in changed:
    target=out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/p,target)
    repair+=''.join(difflib.unified_diff((v1/'source'/p).read_text().splitlines(True),(repo/p).read_text().splitlines(True),fromfile='a/'+p,tofile='b/'+p))
(out/'repair.patch').write_text(repair)
patch=git('diff','--binary')
new=git('ls-files','--others','--exclude-standard').decode().splitlines()
for p in new:
    r=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],cwd=repo,stdout=subprocess.PIPE);assert r.returncode==1;patch+=r.stdout
(out/'b5.patch').write_bytes(patch)
for name in ('b5.patch','repair.patch'):subprocess.run(['git','apply','--reverse','--check',str(out/name)],cwd=repo,check=True)
subprocess.run(['git','diff','--check'],cwd=repo,check=True)
(out/'logs/diff-check.log').write_text('git diff --check: exit 0\ngit apply --reverse --check v2/b5.patch: exit 0\ngit apply --reverse --check v2/repair.patch: exit 0\n')
index={p:{**d,'sha256':sha(repo/p),'bytes':(repo/p).stat().st_size,'source_snapshot':str((out if p in changed else v1)/'source'/p)} for p,d in oldindex.items()}
put('source-index.json',index);put('dependency-index.json',{p:sha(repo/p) for p in olddeps})
put('coverage-gap-reuse.json',{'repair':'B5-R1','v1_freeze':{'path':str(v1/'freeze.json'),'sha256':sha(v1/'freeze.json')},'independent_review':{'path':str(v1/'independent-review.json'),'sha256':sha(v1/'independent-review.json')},'new_evidence':{'red':'logs/red.log: 5 failed,8 passed; four actual changed terminal reuse defects plus true missing new workspace','intermediate':'logs/green.log:13 passed; superseded by final terminal primitive tail','newly_identified_gap':'logs/inf-component.log:2 passed,1 failed; unchanged +Inf normalSpeed actual fresh projection tail','final':'logs/final.log:16 passed,8 deselected; actual closure call counts and complete layer/trace/packet fields'},'reused':{'v1_evidence_artifacts':len(oldartifacts),'verified_hash_mismatches':0,'v1_dependency_entries':len(olddeps),'changed_dependencies':depschanged,'unchanged_source_files':len(oldindex)-len(changed),'original_success_traces':'11/29/150 + analytic overflow unchanged; no rebuild','cuda':'two v1 real nvcc sm80 receipts unchanged; neither TU includes GasWallMath; no unrelated recompile','v1_independent_review':'All findings except R1 retained; B5-R1 awaits scoped re-review'},'remaining_gaps':['Independent R1 re-review','B6 lifecycle/reconfiguration/ownership','B7 stale stage actual producer + dependent face scheduling','B7 HostState terminal material sampling','B7 global status/retry/commit consumption','Native GPU/OF execution and performance unavailable','Unified PR14 final acceptance; no merge'], 'no_broad_rerun':True})
put('field-consumer-cost.json',{'owner':expected[0],'fields':'prepared vs current velocity components; actual terminal closure.trace.velocity; downstream primary/pore momentum and energy; layer traction/heat/species reused only when matching','ordinary_finite':'Original mag(delta)>1e-11*max(1,mag(current)) expression unchanged. Only constant finite classifications added; zero extra layer calls.','changed_terminal':'Mismatch follows existing real closure once; no numeric refusal.','matching_terminal':'Same component equality or both NaN; zero layer calls; finish the two existing fresh velocity projections using already evaluated mass/rho, no second property pass.','true_missing':'Fresh closure with no actual required workspace returns false; matching prepared-only still succeeds without workspace.','precision':'unchanged double','runtime_storage_and_work':'No allocation, transfer, persistent snapshot, availability array, cache, whole-species scan, global traversal or synchronization.','gate_review':'No new reject branch. Numerical classifications choose reuse vs actual work only. Existing real unavailable fresh closure return retained.'})
commands=[]
for stage,selection in [('red','prepared_velocity'),('green','prepared_velocity'),('inf-component','inf_speed'),('final','prepared_velocity')]:
    commands.append({'argv':['python','-m','pytest','-q',expected[1],'-k',selection,'--basetemp='+str(out/'tmp'/stage)],'log':'logs/'+stage+'.log','compiler_and_input':'tmp/'+stage+'/prepared-velocity0/{command.txt,reuse.cpp,build.log,reuse,run-*.log}'})
put('commands.json',{'cwd':str(repo),'environment':{'PYTHONPATH':str(repo.parent/'chmt_remaining_audit_20261013/python_deps'),'TMPDIR':str(out/'tmp')},'executed':commands,'compiler_flags':'g++ C++17 -O0 -ffp-contract=off -finstrument-functions -fno-inline -Wall -Wextra -Werror; command.txt records exact invocation','no_cuda_rerun':True})
artifacts={}
for p in sorted(out.rglob('*')):
    if not p.is_file() or p.is_symlink() or p.name in ('evidence-index.json','freeze.json'):continue
    artifacts[str(p.relative_to(out))]={'sha256':sha(p),'bytes':p.stat().st_size}
put('evidence-index.json',artifacts)
put('freeze.json',{'batch':'B5-v2','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':git('rev-parse','HEAD').decode().strip(),'branch':git('branch','--show-current').decode().strip(),'production_changed_from_v1':[expected[0]],'tests_changed_from_v1':[expected[1]],'full_patch_sha256':sha(out/'b5.patch'),'repair_patch_sha256':sha(out/'repair.patch'),'report_sha256':sha(out/'report.md'),'source_index_sha256':sha(out/'source-index.json'),'dependency_index_sha256':sha(out/'dependency-index.json'),'evidence_index_sha256':sha(out/'evidence-index.json'),'new_artifact_count':len(artifacts),'reused_artifact_count':len(oldartifacts),'v1_freeze_sha256':sha(v1/'freeze.json'),'review_status':'PENDING same-reviewer B5-R1 re-review','new_tests':'16 passed,8 unrelated deselected','CUDA':'reuse two valid v1 compiler receipts; no GPU run','no_commit_push':True,'no_README':True,'no_new_common_file':True,'remaining_scope':'B6/B7 and final PR14 acceptance remain'})
print(json.dumps({'v2_patch_sha256':sha(out/'b5.patch'),'repair_sha256':sha(out/'repair.patch'),'freeze_sha256':sha(out/'freeze.json'),'new_artifacts':len(artifacts),'reused':len(oldartifacts),'changed_deps':depschanged},indent=2))
