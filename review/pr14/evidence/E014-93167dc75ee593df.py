from pathlib import Path
import hashlib,json,shlex,difflib
out=Path(__file__).resolve().parent;parent=out.parent;root=out.parents[3]/'chmt_remaining_work_20261013'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for folder in [out,parent/'pool-delta']:
 indexed=[]
 for dep in (folder/'cuda').glob('*.d'):
  items=dep.read_text().split(':',1)[1].replace('\\\n',' ').split();entries=[]
  for text in items:
   p=Path(text).resolve();entries.append({'path':str(p),'sha256':sha(p)})
   if p.is_relative_to(root):
    target=folder/'inputs'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
  indexed.append({'depfile':str(dep.relative_to(folder)),'dependencies':entries})
 (folder/'compiler-dependencies.json').write_text(json.dumps(indexed,indent=2))
folder=parent/'pool-delta';entries=[];patch=''
for name in ['common/GpuPoolParticleContribution.inl','tests/test_particle_routing_continuation.py']:
 old=folder/'before'/name;p=root/name;t=folder/'source'/name;t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(p.read_bytes())
 entries.append({'path':name,'before_sha256':sha(old),'sha256':sha(p)})
 patch+='diff --git a/'+name+' b/'+name+'\n'+''.join(difflib.unified_diff(old.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
(folder/'delta.patch').write_text(patch);(folder/'source-manifest.json').write_text(json.dumps(entries,indent=2))
old=json.loads((parent/'source-manifest.json').read_text());allpaths=[r['path'] for r in old]+['common/GpuPoolParticleContribution.inl']
(folder/'combined-owned-source-manifest.json').write_text(json.dumps([{'path':name,'sha256':sha(root/name)} for name in allpaths],indent=2))
