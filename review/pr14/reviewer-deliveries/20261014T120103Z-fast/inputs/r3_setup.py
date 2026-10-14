from pathlib import Path
import json,subprocess,os,datetime,hashlib,shutil
P=Path(__file__).resolve().parent;incoming=json.loads((P/'r3_incoming.json').read_text());O=Path(incoming['old_checkout']);area=Path(incoming['area']);source=incoming['source'];head=incoming['remote_head']
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
def g(*args):return subprocess.check_output(['git','-C',str(O),*args],text=True,env=env).strip()
assert g('rev-parse','origin/perf/chmt-first-round-20261013')==head
batch=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-fast');R=area/'pr-review'/('pr14-'+batch);D=R/'review/pr14/reviewer-deliveries'/batch
assert R.resolve().is_relative_to(area.resolve()) and not R.exists()
q=subprocess.run(['git','-C',str(O),'worktree','add','-b','reviewer/pr14-'+batch,str(R),head],capture_output=True,text=True,env=env,timeout=120);assert q.returncode==0,q.stderr
for name in ['inputs','logs','artifacts','metadata','patches']:(D/name).mkdir(parents=True)
(R/'work').mkdir()
data={'batch':batch,'repo':str(area),'checkout':str(R),'delivery':str(D),'source':source,'source_tree':incoming['source_tree'],'delivery_parent':head,'old_checkout':str(O),'old_delivery':incoming['old_delivery'],'old_source':incoming['old_source'],'old_delivery_commit':incoming['old_delivery_commit'],'baseline':'d6110c02745a8ad726208a426c4bb3a28233c1fd','developer_directory':incoming['developer_directory'],'branch':'reviewer/pr14-'+batch,'prior_round1_source':'802a11e7dfe986488a8bf8305e86424dcc94c9e1','prior_round1_evidence':'ae5a59478023c74e23b64dc1a1f388fb805ef565','identity':'PR14_REVIEW_ONLY','interaction_class':'FAST'}
(P/'r3_paths.json').write_text(json.dumps(data,indent=2)+'\n');(D/'SOURCE.json').write_text(json.dumps(data,indent=2)+'\n');(R/'work/paths.json').write_text(json.dumps(data,indent=2)+'\n')
oldD=Path(incoming['old_delivery']);oldenv=oldD/'inputs/env.sh'
if oldenv.is_file():content=oldenv.read_bytes()
else:
 ref=next(q for q in json.loads((oldD/'metadata/immutable-references.json').read_text())['references'] if q['alias']=='inputs/env.sh')
 content=subprocess.check_output(['git','-C',str(O),'show',ref['fixed_commit']+':'+ref['repo_path']]);assert hashlib.sha256(content).hexdigest()==ref['sha256']
# The generic environment wrapper derives all installations from its caller cwd.
text=content.decode();assert '/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal/' not in text
(D/'inputs/env.sh').write_text(text.replace(str(O),str(R)))
sources=[]
for line in g('ls-tree','-r',source,'applications','common','tests','tools').splitlines():
 meta,path=line.split('\t',1);mode,typ,blob=meta.split();p=R/path
 if p.is_file():sources.append({'path':path,'blob':blob,'mode':mode,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(D/'metadata/source-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
(D/'metadata/formal-production-and-support.diff').write_text(g('diff',incoming['old_source'],source,'--','applications','common','tests','tools')+'\n')
(D/'results.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':source,'source_tree':incoming['source_tree'],'record_parent':head,'tests':[]},indent=2)+'\n')
(D/'commands.txt').write_text('PR14_REVIEW_ONLY round3 FAST; actual executed commands only.\nDefault run300s; explicitly planned Ns10 reacting initcheck1800s and originalevap900s.\n')
print(json.dumps(data,indent=2));print('ENV\n'+(D/'inputs/env.sh').read_text())
for name in ['REPORT.md','SOURCE.json','EVIDENCE_REUSE.md','GAP_RECHECK_PLAN.md']:
 p=R/incoming['developer_directory']/name;print('\nFILE',name,'\n'+p.read_text())
v=json.loads((R/incoming['developer_directory']/'VALIDATION.json').read_text());print('VALIDATION_KEYS',v.keys());print(json.dumps(v,indent=2))
