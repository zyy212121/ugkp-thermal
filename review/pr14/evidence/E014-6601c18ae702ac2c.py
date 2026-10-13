from pathlib import Path
import hashlib,json,subprocess,ast,re
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b4/v2');old=root.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();f=json.loads((root/'freeze.json').read_text());v1=json.loads((old/'freeze.json').read_text());idx=json.loads((root/'evidence-index.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==f['head']
assert sha(root/'b4.patch')==f['complete_patch_sha256']=='6f96bdddda8ae5974635822e320494e332b1e8c12e5046a6187d324c7ec0b3ca'
for name,key in [('repair.patch','repair_patch_sha256'),('report.md','report_sha256'),('evidence-index.json','evidence_index_sha256')]:assert sha(root/name)==f[key]
assert sha(old/'freeze.json')==f['v1_freeze_sha256']
assert sha(old/'independent-review.json')==f['review_sha256']
assert sha(old/'evidence-index.json')==v1['evidence_index_sha256']
for x in idx:assert sha(root/x['path'])==x['sha256'] and (root/x['path']).stat().st_size==x['size'],x['path']
changed=[]
for x in f['files']:
 assert sha(repo/x['path'])==sha(root/'source'/x['path'])==x['sha256']
 assert sha(old/'source-final'/x['path'])==x['v1_sha256']
 if x['sha256']!=x['v1_sha256']:changed.append(x['path'])
assert changed==f['changed_from_v1']
patch=subprocess.check_output(['git','diff','--binary'],cwd=repo)+subprocess.run(['git','diff','--no-index','--binary','--','/dev/null','tests/chemistry/test_gas_terminal_completion.py'],cwd=repo,stdout=subprocess.PIPE).stdout
assert patch==(root/'b4.patch').read_bytes()
assert sha(root/'source-before/ConservativeReactor.H')==next(x['sha256'] for x in v1['files'] if x['path']=='common/chemistry/ConservativeReactor.H')
for x in json.loads((root/'commands.json').read_text()):assert hashlib.sha256(x['command'].encode()).hexdigest()==x['command_sha256'] and sha(root/x['output'])==x['output_sha256']
def source_expr(path):
 return next(n.value for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE' for t in n.targets))
rates=ast.literal_eval(source_expr(repo/'tests/chemistry/test_gas_rates.py'))
regressions=rates.split('int main(){')[0]+ast.literal_eval(source_expr(repo/'tests/chemistry/test_gas_review_regressions.py').right)
for kind in ['red','green']:
 found=list({p.resolve() for p in (root/'tmp'/kind).glob('gas-review-regressions*/probe.cpp')});assert len(found)==1;assert found[0].read_text()==regressions
receipt=json.loads((root/'paired-receipt.json').read_text());assert sha(root/'receipts/zero-rank-unused-caloric.cpp')==sha(old/'review/zero-rank-unused-caloric.cpp')==receipt['source_sha256']
assert sha(root/'receipts/zero-rank-v2')==receipt['executable_sha256']
assert sha(root/'logs/paired-v2.log')==receipt['stdout_sha256']
assert sha(Path(receipt['reused_original_stdout']))==receipt['reused_original_stdout_sha256']
norm=lambda s:re.sub(r' available=[01]','',s)
assert norm((root/'logs/paired-v2.log').read_text())==norm(Path(receipt['reused_original_stdout']).read_text())
assert all('available=1' in line for line in (root/'logs/paired-v2.log').read_text().splitlines())
assert sha(root/'cuda/chemistry.cu')==f['cuda']['source_sha256'] and sha(root/'cuda/chemistry.o')==f['cuda']['object_sha256'] and sha(root/'logs/cuda.log')==f['cuda']['output_sha256']
assert (root/'cuda/chemistry.cu').read_bytes()==(old/'cuda/chemistry.cu').read_bytes()
assert (root/'cuda/chemistry.o').read_bytes()[:4]==b'\x7fELF'
check=subprocess.run(['git','diff','--check'],cwd=repo,capture_output=True);assert check.returncode==0
reverse=subprocess.run(['git','apply','--reverse','--check',str(root/'b4.patch')],cwd=repo,capture_output=True);assert reverse.returncode==0
print(json.dumps({'indexed_v2_artifacts_verified':len(idx),'frozen_files':len(f['files']),'changed_from_v1':changed,'original_review_and_evidence_anchors':'unchanged','generated_red_green_cpp':'matches frozen current test','paired_three_mode_comparison':'shared fields identical; all v2 available true','cuda_object':'hash and ELF verified; no execution','diff_check':check.returncode,'reverse_apply_check':reverse.returncode,'new_test_or_compiler_execution':'NONE'},indent=2))
