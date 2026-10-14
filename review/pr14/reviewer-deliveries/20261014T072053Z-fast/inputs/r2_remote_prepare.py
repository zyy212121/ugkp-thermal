from r2_runner import *
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
p=subprocess.run(['git','-C',str(ROOT),'fetch','origin','perf/chmt-first-round-20261013'],capture_output=True,text=True,env=env);print('fetch',p.returncode,p.stdout,p.stderr)
remote=git('rev-parse','FETCH_HEAD');record={'checked_utc':utc(),'remote_head':remote,'review_start_parent':PATHS['delivery_parent'],'reviewed_source':SOURCE,'remote_change_since_review_start':remote!=PATHS['delivery_parent'],'production_source_diff_after_start':git('diff','--stat',SOURCE,remote,'--','applications','common','tests','tools'),'status':git('status','--short'),'fetch_exit':p.returncode};(D/'metadata/prepublication-remote-state.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
print('results',len(json.loads((D/'results.json').read_text())['tests']))
