from pathlib import Path
import subprocess, json, hashlib, datetime, os, shutil

P=Path(__file__).resolve().parent
incoming=json.loads((P/'r4_incoming.json').read_text())
area=Path(incoming['area']).resolve()
old=Path(incoming['old_checkout'])
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
def g(*a): return subprocess.check_output(['git','-C',str(old),*a],env=env).strip()
def digest(b): return hashlib.sha256(b).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc)
batch=now.strftime('%Y%m%dT%H%M%SZ')+'-fast'
root=area/'pr-review'/('pr14-'+batch)
assert root.resolve().is_relative_to(area) and not root.exists()
parent=incoming['remote_head'];source=incoming['source']
assert not g('diff','--name-only',source,parent,'--','applications','common','tests','test','tools')
subprocess.run(['git','-C',str(old),'worktree','add','-b','reviewer/pr14-'+batch,str(root),parent],env=env,check=True)
d=root/'review/pr14/reviewer-deliveries'/batch
for n in ['metadata','logs','inputs','artifacts','patches','fixtures']:(d/n).mkdir(parents=True,exist_ok=True)
(root/'work').mkdir()
paths={**incoming,'batch':batch,'repo':str(area),'checkout':str(root),'delivery':str(d),'delivery_parent':parent,'branch':'reviewer/pr14-'+batch}
(P/'r4_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
(d/'SOURCE.json').write_text(json.dumps({**paths,'recorded_utc':now.isoformat(),'production_run_started':False,'merge_authorized':False,'publication_status':'PENDING'},indent=2)+'\n')
dev=root/incoming['developer_directory']
listing=[];mismatch=[]
for line in (dev/'SHA256SUMS').read_text().splitlines():
 if not line.strip():continue
 expected,name=line.split(None,1);name=name.lstrip('*').strip();path=dev/name
 actual=digest(path.read_bytes()) if path.is_file() else None
 listing.append({'path':name,'expected_sha256':expected,'actual_sha256':actual,'match':actual==expected})
 if actual!=expected:mismatch.append(name)
tracked=g('ls-tree','-r','--name-only',parent,incoming['developer_directory']).decode().splitlines()
manifest_names={r['path'] for r in listing}
unlisted=[n[len(incoming['developer_directory'])+1:] for n in tracked if n[len(incoming['developer_directory'])+1:] not in manifest_names]
remote_receipt={'identity':'PR14_REVIEW_ONLY','recorded_utc':now.isoformat(),'remote_fixed_commit':parent,'directory':incoming['developer_directory'],'tracked_files':len(tracked),'checksum_entries':len(listing),'mismatches':mismatch,'unlisted_files':unlisted,'checks':listing,'verification':'all listed bytes read from fetched fixed remote tree and checked against uploaded SHA256SUMS; source remains separately fixed','developer_publication_flags_unmodified':True}
(d/'metadata/developer-remote-evidence-verification.json').write_text(json.dumps(remote_receipt,indent=2)+'\n')
assert not mismatch,mismatch
claimed=json.loads((dev/'SOURCE.json').read_text())['changed_files']
changes=g('diff','--name-only',incoming['old_delivery_commit'],source).decode().splitlines()
# Source139 is the exact non-review source delta, not the publication/document delta.
changes=[p for p in changes if not p.startswith('review/')]
assert set(changes)=={r['path'] for r in claimed},(len(changes),len(claimed))
source_rows=[]
for q in claimed:
 path=q['path'];actual=None;blob=None
 if not q['deleted']:
  payload=g('show',source+':'+path);actual=digest(payload)
  # git show is binary-safe; do not strip the terminal newline when hashing.
  payload=subprocess.check_output(['git','-C',str(old),'show',source+':'+path]);actual=digest(payload)
  blob=g('rev-parse',source+':'+path).decode();assert actual==q['sha256'],path
 else:assert not (root/path).exists(),path
 source_rows.append({**q,'computed_sha256':actual,'blob':blob,'match':actual==q['sha256']})
(d/'metadata/source139-verification.json').write_text(json.dumps({'source_commit':source,'tree':incoming['source_tree'],'baseline':incoming['old_delivery_commit'],'exact_changed_path_set':True,'entries':source_rows},indent=2)+'\n')
rows=[]
for path in g('ls-tree','-r','--name-only',source,'applications','common','tests','test','tools').decode().splitlines():
 p=root/path
 if p.is_file():rows.append({'path':path,'blob':g('rev-parse',source+':'+path).decode(),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size})
(d/'metadata/source-hashes.json').write_text(json.dumps(rows,indent=2)+'\n')
(d/'metadata/production-diff.patch').write_bytes(subprocess.check_output(['git','-C',str(old),'diff',incoming['old_source'],source,'--','applications','common','tests','tools']))
(d/'metadata/environment-initial.txt').write_text('\n'.join(subprocess.run(a,capture_output=True,text=True).stdout for a in [['nvidia-smi'],['free','-b'],['uname','-a'],['findmnt','-T',str(area),'-o','TARGET,SOURCE,FSTYPE,OPTIONS']]))
(d/'results.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':source,'source_tree':incoming['source_tree'],'tests':[],'acceptance_units':[]},indent=2)+'\n')
(d/'commands.txt').write_text('PR14_REVIEW_ONLY FAST4\nAll automatic elapsed-time termination disabled by explicit user instruction.\n')
envscript='''#!/usr/bin/env bash
set -o pipefail
source /opt/openfoam10/etc/bashrc
set -e
ROOT="$(git -C "$(dirname "$0")/../../../.." rev-parse --show-toplevel)"
export CUDA_HOME=/usr/local/cuda
export CUDA_ARCH=sm_89
export PATH="$CUDA_HOME/bin:$PATH"
export FOAM_USER_APPBIN="$ROOT/work/install/ns2/bin"
export FOAM_USER_LIBBIN="$ROOT/work/install/ns2/lib"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=2
mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"
exec "$@"
'''
(d/'inputs/env.sh').write_text(envscript)
for name in ['r4_setup.py','r4_paths.json']:shutil.copy2(P/name,d/'inputs'/name)
print(json.dumps({'checkout':str(root),'delivery':str(d),'source139_verified':len(source_rows),'remote_checksum_entries':len(listing),'unlisted':unlisted,'source_files':len(rows)},indent=2),flush=True)
