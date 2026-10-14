import os, pathlib,subprocess,datetime,json,hashlib,shutil
P=pathlib.Path(__file__).parent
OLDROOT=pathlib.Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr-review/pr14-20261013T182722Z-fast')
HEAD='75cfd885c028fe80d011a8a081f8bed1f8aeb55e'
SOURCE='4b5c480b572f234ddecf5e89164fe59b0d2e9735'
batch=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-fast')
R=OLDROOT.parent/('pr14-'+batch);D=R/'review/pr14/reviewer-deliveries'/batch
def g(*a):return subprocess.check_output(['git','-C',str(OLDROOT),*a],text=True).strip()
assert not g('diff',SOURCE,HEAD,'--','applications','common','tests','tools'), 'Head production differs from SOURCE'
subprocess.run(['git','-C',str(OLDROOT),'worktree','add','-b','reviewer/pr14-'+batch,str(R),HEAD],env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1'),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
for p in ['inputs','artifacts','logs','patches','metadata']: (D/p).mkdir(parents=True)
(R/'work').mkdir()
data={'batch':batch,'repo':str(OLDROOT.parent.parent),'checkout':str(R),'delivery':str(D),'old_checkout':str(OLDROOT),'old_delivery':str(OLDROOT/'review/pr14/reviewer-deliveries/20261013T182722Z-fast'),'source':SOURCE,'source_tree':g('rev-parse',SOURCE+'^{tree}'),'delivery_parent':HEAD,'old_source':'802a11e7dfe986488a8bf8305e86424dcc94c9e1','old_delivery_commit':'ae5a59478023c74e23b64dc1a1f388fb805ef565','baseline':'d6110c02745a8ad726208a426c4bb3a28233c1fd','developer_directory':'review/pr14/developer-deliveries/20261014T055427Z-fast','branch':'reviewer/pr14-'+batch}
(P/'r2_paths.json').write_text(json.dumps(data,indent=2)+'\n');(R/'work/paths.json').write_text(json.dumps(data,indent=2)+'\n')
shutil.copy2(OLDROOT/'review/pr14/reviewer-deliveries/20261013T182722Z-fast/inputs/env.sh',D/'inputs/env.sh')
sources=[]
for line in g('ls-tree','-r',SOURCE,'applications','common','tests','tools').splitlines():
 meta,path=line.split('\t',1);mode,typ,blob=meta.split();p=R/path
 if p.is_file():sources.append({'path':path,'blob':blob,'mode':mode,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(D/'metadata/source-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
(D/'SOURCE.json').write_text(json.dumps(data,indent=2)+'\n')
(D/'results.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'tests':[]},indent=2)+'\n')
(D/'commands.txt').write_text('PR14_REVIEW_ONLY round 2 FAST\nExecuted commands only; 300s default external case limit. Native builds have 900s cap due prior measured compilation.\n')
diff=g('diff','802a11e7',SOURCE,'--','applications','common','tests');(D/'metadata/formal-production-and-support.diff').write_text(diff+'\n')
print(json.dumps(data,indent=2))
dev=R/data['developer_directory'];v=json.loads((dev/'VALIDATION.json').read_text());print('VALIDATION COMPLETE READ scopes',[(x.get('name'),x.get('exit_code'),x.get('artifact_sha256')) for x in v.get('scopes',[])], 'keys',list(v))
print('VALIDATION OTHER',json.dumps({k:v[k] for k in v if k!='scopes'},indent=2))
old=pathlib.Path(data['old_delivery']);c=json.loads((old/'coverage.json').read_text());print('OLD COVERAGE KEYS',list(c))
for k in c:
 if 'gap' in k.lower() or 'not_run' in k.lower(): print(k,json.dumps(c[k],indent=2))
for name in ['REPORT.md','findings.json']:
 text=(old/name).read_text();print('OLD '+name+'\n'+text)
res=json.loads((old/'results.json').read_text());print('OLD RESULTS SUMMARY',len(res['tests']),[(x['id'],x['verdict']) for x in res['tests'] if x['verdict']!='PASS'])
