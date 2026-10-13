from pathlib import Path
import subprocess,hashlib,json,datetime,shutil
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b4');base='cf49400267b0df39017d6ce348ce72cbcab6b513'
def sha(data):return hashlib.sha256(data).hexdigest()
files=subprocess.check_output(['git','diff','--name-only'],cwd=repo,text=True).splitlines()+['tests/chemistry/test_gas_terminal_completion.py'];records=[]
for rel in files:
 p=repo/rel;original=subprocess.run(['git','show',base+':'+rel],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 records.append({'path':rel,'sha256':sha(p.read_bytes()),'base_sha256':sha(original.stdout) if original.returncode==0 else None})
 q=root/'source-final'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
patch=subprocess.check_output(['git','diff','--binary'],cwd=repo)+subprocess.run(['git','diff','--no-index','--binary','--','/dev/null','tests/chemistry/test_gas_terminal_completion.py'],cwd=repo,stdout=subprocess.PIPE).stdout
(root/'b4.patch').write_bytes(patch)
check=subprocess.run(['git','diff','--check'],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(root/'logs/diff-check.log').write_bytes(check.stdout);assert check.returncode==0
# Each probe source, command runner, retained executable/object, comparison
# input/output and failed stdout is individually hashed. Symlinks only alias
# pytest's already-indexed directories, so are not followed as new evidence.
index=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink() or p.name in ['freeze.json','evidence-index.json']:continue
 index.append({'path':str(p.relative_to(root)),'size':p.stat().st_size,'sha256':sha(p.read_bytes())})
(root/'evidence-index.json').write_text(json.dumps(index,indent=2)+'\n')
refs=[]
for batch,version in [('b1','v2'),('b2','v3'),('b3','v2')]:
 for name in ['report.md','freeze.json','independent-review.md','independent-review.json']:
  p=root.parent/batch/version/name
  if p.exists():refs.append({'path':str(p),'sha256':sha(p.read_bytes())})
freeze={'schema':1,'batch':'B4','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':base,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),'files':records,'patch_sha256':sha(patch),'report_sha256':sha((root/'report.md').read_bytes()),'evidence_index_sha256':sha((root/'evidence-index.json').read_bytes()),'field_consumer_cost_sha256':sha((root/'field-consumer-cost.json').read_bytes()),'coverage_gap_reuse_sha256':sha((root/'coverage-gap-reuse.json').read_bytes()),'inherited_independent_evidence':refs,'original_lineage_commit':'d6110c02745a8ad726208a426c4bb3a28233c1fd','accuracy_receipt':'logs/accuracy.log','focused_receipt':'final-run.json','final_delta_receipt':'final-delta-run.json','final_troe_receipt':'logs/troe-final.log','final_cuda':{'source':'cuda/chemistry.cu','object':'cuda/chemistry-final.o','log':'logs/cuda-final.log','command':['/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/toolchains/cuda/bin/nvcc','-std=c++17','-arch=sm_80','--expt-relaxed-constexpr','-I.','-c',str(root/'cuda/chemistry.cu'),'-o',str(root/'cuda/chemistry-final.o')],'exit_code':0,'gpu_execution':'NOT_RUN'},'notes':['No new common paths; all eight owners already registered.','No README, commit, push, merge, or B5–7 broad production edits.','Exact historical twelve receipts unavailable; explicit available twelve executed, no historical mapping invented.','B7 actual GPU/outer status/half-capture consumers remain pending.']}
(root/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps({'files':len(records),'patch_sha256':freeze['patch_sha256'],'report_sha256':freeze['report_sha256'],'indexed_evidence_files':len(index),'head':freeze['head']},indent=2))
