from pathlib import Path
import json,subprocess,os,datetime
P=Path(__file__).resolve().parent;prev=json.loads((P/'r3_paths.json').read_text());R=Path(prev['checkout']);area=Path(prev['repo']);source='47034e4e0380fa5b84165b49207a95a9b187a52a';expected='c932823a00b3ed2f647e22de5002f9b913962fee';env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
def g(*a):return subprocess.check_output(['git','-C',str(R),*a],text=True,env=env).strip()
q=subprocess.run(['git','-C',str(R),'fetch','origin','perf/chmt-first-round-20261013'],capture_output=True,text=True,env=env);print('FETCH',q.returncode,q.stdout,q.stderr);assert q.returncode==0
head=g('rev-parse','origin/perf/chmt-first-round-20261013');tree=g('rev-parse',source+'^{tree}');assert tree==expected
data={'area':str(area),'old_checkout':str(R),'old_delivery':prev['delivery'],'old_source':prev['source'],'old_delivery_commit':'0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668','source':source,'source_tree':tree,'remote_head':head,'developer_directory':'review/pr14/developer-deliveries/20261014T180000Z-fast','identity':'PR14_REVIEW_ONLY','interaction_class':'FAST'}
(P/'r4_incoming.json').write_text(json.dumps(data,indent=2)+'\n');print('IDENTITY',data)
print('SOURCE_CHANGES',g('diff','--stat',prev['source'],source,'--','applications','common','tests','tools'))
print('REMOTE_CODE_DELTA',g('diff','--name-only',source,head,'--','applications','common','tests','tools'));print('LOCAL_WORKTREES',g('worktree','list','--porcelain'))
paths=g('ls-tree','-r','--name-only',head,data['developer_directory']).splitlines();print('REMOTE_DEVELOPER_FILES',len(paths));print('\n'.join(paths[:25]));
for name in ['REPORT.md','SOURCE.json','VALIDATION.json','EVIDENCE_REUSE.md','GAP_RECHECK_PLAN.md']:
 path=data['developer_directory']+'/'+name
 try:text=subprocess.check_output(['git','-C',str(R),'show',head+':'+path],text=True)
 except subprocess.CalledProcessError:continue
 (P/('r4-dev-'+name.replace('.','_'))).write_text(text);print('FILE',name,'\n'+text)
print('AGENTS',*[str(p) for p in [area/'AGENTS.md',area.parent/'AGENTS.md',area.parent.parent/'AGENTS.md'] if p.is_file()])
print('NVIDIA',subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
print('MEMINFO',Path('/proc/meminfo').read_text()[:650]);print('MOUNT',subprocess.run(['findmnt','-T',str(area),'-o','TARGET,SOURCE,FSTYPE,OPTIONS'],capture_output=True,text=True).stdout)
print('RUNNING_NVCC_CHMT',subprocess.run(['ps','-eo','pid,ppid,comm,args'],capture_output=True,text=True).stdout[-10000:])
