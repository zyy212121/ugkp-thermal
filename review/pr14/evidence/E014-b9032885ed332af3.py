from pathlib import Path
import subprocess,json,hashlib,difflib,datetime
R=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');O=Path(__file__).parent;V1=O.parent
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=lambda *a:subprocess.check_output(a,cwd=R,text=True)
files=sorted(set(run('git','diff','--name-only').splitlines()+run('git','ls-files','--others','--exclude-standard').splitlines()))
changed=[];repair=''
for name in files:
 old=O/'v1-source'/name
 assert old.is_file(),name
 if old.read_bytes()!=(R/name).read_bytes():
  changed.append(name);repair+=''.join(difflib.unified_diff(old.read_text().splitlines(keepends=True),(R/name).read_text().splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
(O/'repair.patch').write_text(repair)
patch=subprocess.check_output(['git','diff','--binary'],cwd=R)
for name in run('git','ls-files','--others','--exclude-standard').splitlines():
 r=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],cwd=R,capture_output=True);assert r.returncode==1;patch+=r.stdout
(O/'full.patch').write_bytes(patch)
for name in ('repair.patch','full.patch'):
 r=subprocess.run(['git','apply','--check','--reverse',str(O/name)],cwd=R,capture_output=True,text=True);(O/'logs'/f'{name}-check.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
old_index=json.loads((V1/'artifact-index.json').read_text());bad=[x['path']for x in old_index if h(V1/x['path'])!=x['sha256']];assert not bad,bad
oldfreeze=json.loads((V1/'freeze.json').read_text())
reused=[{'path':x['path'],'sha256':x['sha256']} for x in oldfreeze['sources']if h(R/x['path'])==x['sha256']]
receipts=['logs/red.log','logs/green.log','logs/tails-final.log','logs/compound-allocation-final.log','logs/affected.log','logs/cost-compare.log','cuda/commands.json','logs/diff-check.log','logs/managed.log']
reuse={'v1_freeze_sha256':h(V1/'freeze.json'),'v1_index_sha256':h(V1/'artifact-index.json'),'v1_all_indexed_artifacts_hash_verified':len(old_index),'v1_independent_review_sha256':h(V1/'independent-review.md'),'unchanged_v1_sources':reused,'v2_changed_from_v1':changed,'reuse':'Independent v1 successful physical formulas, first-success bits/ordering, CHMT lifecycle, SST constants and graph capture evidence reused. Native fixed whole-state overhead superseded by 376-byte owner metadata disclosure; native wall/snapshot/budget tails have fresh actual host-CUDA cases and actual enabled-budget nvcc instantiation. Passing executions are non-additive.','new_receipts':[{'path':p,'sha256':h(O/p)}for p in receipts],'gaps':['Independent R1/R2 re-review pending','B7/global size veto, current stage scheduling, material sampling and status/commit consumption remain','No GPU/asynchronous runtime stress or OF integration/performance','Terminal releaseState retains its existing void best-effort destruction API; persistent free failures during final destroy cannot be retried after caller discards handle','Peer mirror repo unavailable']}
(O/'reuse-gap.json').write_text(json.dumps(reuse,indent=2)+'\n')
index=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and not p.is_symlink() and p.name not in ('artifact-index.json','freeze.json'):
  index.append({'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':h(p)})
(O/'artifact-index.json').write_text(json.dumps(index,indent=2)+'\n')
f={'scope':'B6-R1/R2 v2 candidate for independent re-review; no commit/push/merge/GPU/whole-PR claim','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':run('git','rev-parse','HEAD').strip(),'branch':run('git','branch','--show-current').strip(),'v1_freeze_sha256':h(V1/'freeze.json'),'v2_delta_files':changed,'sources':[{'path':n,'sha256':h(R/n),'bytes':(R/n).stat().st_size}for n in files],'repair_patch_sha256':h(O/'repair.patch'),'full_patch_sha256':h(O/'full.patch'),'report_sha256':h(O/'report.md'),'reuse_gap_sha256':h(O/'reuse-gap.json'),'artifact_index_sha256':h(O/'artifact-index.json'),'artifacts':len(index),'v1_verified_artifacts':len(old_index),'readme_changed':any(Path(n).name.lower().startswith('readme')for n in files),'diff_check':True,'patch_reverse_checks':True};assert not f['readme_changed'];(O/'freeze.json').write_text(json.dumps(f,indent=2)+'\n');print('FROZEN',len(files),'total sources',len(changed),'v2 delta files',len(index),'new indexed artifacts; v1 unchanged',len(old_index));print('REPAIR',f['repair_patch_sha256']);print('FULL',f['full_patch_sha256'])
