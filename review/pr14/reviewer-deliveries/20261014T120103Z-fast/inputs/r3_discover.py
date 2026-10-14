from pathlib import Path
import json,subprocess,os,urllib.request,datetime,hashlib
P=Path(__file__).resolve().parent
old=json.loads((P/'r2_paths.json').read_text());R=Path(old['checkout']);area=Path(old['repo'])
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
def g(*args):return subprocess.check_output(['git','-C',str(R),*args],text=True,env=env).strip()
for p in [area,*area.parents]:
 a=p/'AGENTS.md'
 if a.is_file():print('AGENTS',str(a),a.read_text())
q=subprocess.run(['git','-C',str(R),'fetch','origin','perf/chmt-first-round-20261013'],capture_output=True,text=True,env=env,timeout=120)
assert q.returncode==0,q.stderr
head=g('rev-parse','origin/perf/chmt-first-round-20261013');source='9d5be6eff69dbced2fc959d422d3ba419a1bc39e'
assert g('rev-parse',source+'^{tree}')=='5dc844d4189f50fcfbed2a77fa5e6cd6e9cfd066'
assert not g('diff','--name-only',source,head,'--','applications','common','tests','tools'),'Remote production advanced beyond requested source; inspect causal change before setup.'
for n in g('ls-tree','-r','--name-only',head).splitlines():
 if Path(n).name=='AGENTS.md':print('AGENTS_IN_REPO',n,g('show',head+':'+n))
data={'remote_head':head,'source':source,'source_tree':g('rev-parse',source+'^{tree}'),'old_source':old['source'],'old_delivery_commit':'f5e6c5e2cbde2693f5fa811d874c200b07bbbc66','old_checkout':str(R),'old_delivery':old['delivery'],'area':str(area),'developer_directory':'review/pr14/developer-deliveries/20261014T112400Z-fast','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'r3_incoming.json').write_text(json.dumps(data,indent=2)+'\n')
print('IDENTITY',json.dumps(data,indent=2));print('DIFF',g('diff',old['source'],source,'--','applications','common','tests','tools'))
for name in ['PROTOCOL.md','REPORT.md','SOURCE.json','VALIDATION.json','EVIDENCE_REUSE.md','GAP_RECHECK_PLAN.md']:
 path='review/pr14/'+name if name=='PROTOCOL.md' else data['developer_directory']+'/'+name
 print('\nFILE',path,'\n',g('show',head+':'+path))
print('OLD_FINDINGS',json.loads((Path(data['old_delivery'])/'findings.json').read_text()))
print('OLD_GAPS',json.loads((Path(data['old_delivery'])/'coverage.json').read_text())['remaining_gaps'])
print('OLD_UNITS',[(q['id'],q['verdict'],q['scope'],q.get('limitations')) for q in json.loads((Path(data['old_delivery'])/'results.json').read_text())['acceptance_units']])
