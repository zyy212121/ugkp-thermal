from pathlib import Path
import subprocess,json,hashlib,datetime
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');out=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b7')
def git(*args):return subprocess.check_output(['git',*args],cwd=repo)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
extra={'tests/chemistry/test_gas_chemistry_adapter.py','tests/gas_transport/test_native_interval_continuation.py','tests/gas_transport/test_native_mixture_courant_integration.py','tests/gas_transport/test_mixture_state.py','tests/gas_transport/test_mixture_wall_function.py','tests/gas_transport/test_boundary_layer_state.py'}
new=['applications/CHMT/tests/test_final_gas_consumers.py','tests/gas_transport/test_final_courant_fields.py','tests/gas_transport/test_final_native_consumers.py','tests/gas_transport/test_mixture_selector_fallback.py']
changed=[p for p in git('diff','--name-only').decode().splitlines() if p.startswith(('common/','applications/CHMT/','applications/gasUGKP/')) or p in extra]
assert not any(Path(p).name.lower().startswith('readme') for p in changed+new)
patch=git('diff','--binary','--',*changed)
for p in new:patch+=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',p],cwd=repo,capture_output=True).stdout
(out/'b7.patch').write_bytes(patch)
files=[]
for p in sorted(set(changed+new)):
 try:blob=git('rev-parse','HEAD:'+p).decode().strip()
 except subprocess.CalledProcessError:blob=None
 files.append({'path':p,'base_blob':blob,'sha256':digest(repo/p),'bytes':(repo/p).stat().st_size})
proofs=[]
for path in sorted(out.rglob('*')):
 if path.is_file() and ('logs' in path.parts or path.name in ('report.md','field-status-copy-cost.md','coverage-reuse-gaps.json','measured-costs.json','cold-pinned-comparison.json','receipts.json','final-receipts.json')):
  proofs.append({'path':str(path.relative_to(out)),'sha256':digest(path),'bytes':path.stat().st_size})
freeze={'schema':1,'batch':'B7','base':git('rev-parse','HEAD').decode().strip(),'branch':git('branch','--show-current').decode().strip(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'frozen source for independent review; no commit/push/merge','owned_paths_only':True,'excludes':['tools/**','tools/managed_mirrors.json','unrelated tool worker tests'],'files':files,'patch_sha256':digest(out/'b7.patch'),'artifacts':proofs,'limitations':['No GPU runtime; serial host launches are host evidence only','No giant CUDA TU compilation','Whole PR global audits/tools cleanup remain','No main merge authorization']}
(out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(len(files),'B7 owned files; patch',freeze['patch_sha256'])
