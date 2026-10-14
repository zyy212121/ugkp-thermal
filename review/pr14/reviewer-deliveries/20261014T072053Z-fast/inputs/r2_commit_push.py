from r2_runner import *
import urllib.request
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
branch='perf/chmt-first-round-20261013'; delivery=str(D.relative_to(ROOT));review=ROOT/'review/pr14'
def checked(args,timeout=300):
 q=subprocess.run(args,capture_output=True,text=True,env=env,timeout=timeout)
 assert q.returncode==0,(args,q.returncode,q.stdout[-3000:],q.stderr[-3000:])
 return q
if (ROOT/'work/round2-publication-result.json').exists():
 prior=json.loads((ROOT/'work/round2-publication-result.json').read_text())
 assert prior.get('pushed') and prior.get('commit')==git('rev-parse','HEAD'),'Inspect previous publication before another commit.'
 print('Already pushed',prior['commit']);raise SystemExit(0)
assert not git('diff','--cached','--name-only'),'Unexpected preexisting staged files'
assert git('rev-parse','HEAD')==PATHS['delivery_parent'],'Local base unexpectedly changed'
checked(['git','-C',str(ROOT),'fetch','origin',branch])
before=git('rev-parse','origin/'+branch);assert before==PATHS['delivery_parent'],('Remote advanced; integrate its changes without force before publication.',before)
assert not git('diff','--name-only','HEAD','--','applications','common','tests','tools','README.md')
source=json.loads((D/'metadata/source-hashes.json').read_text())
assert all(sha(ROOT/q['path'])==q['sha256'] for q in source)
count=0
for line in (D/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert sha(D/name)==digest,name;count+=1
checked(['python3',str(review/'validate_records.py')])
checked(['git','-C',str(ROOT),'add','-f','--',delivery])
checked(['git','-C',str(ROOT),'add','--','review/pr14/manifest.json','review/pr14/INDEX.md','review/pr14/pending-findings.json','review/pr14/responses'])
staged=git('diff','--cached','--name-only').splitlines()
allowed=[delivery,'review/pr14/manifest.json','review/pr14/INDEX.md','review/pr14/pending-findings.json','review/pr14/responses']
assert all(any(p==a or p.startswith(a+'/') for a in allowed) for p in staged)
assert not any(pathlib.Path(p).name.lower().startswith('readme') for p in staged)
assert {str(p.relative_to(ROOT)) for p in D.rglob('*') if p.is_file()}.issubset(set(staged))
# Raw logs and verbatim generated files retain their original whitespace.
checked(['git','-C',str(ROOT),'diff','--cached','--check','--','review/pr14/INDEX.md','review/pr14/manifest.json','review/pr14/pending-findings.json','review/pr14/responses',delivery+'/REPORT.md',delivery+'/coverage.json',delivery+'/findings.json'])
message=ROOT/'work/round2-commit-message.txt'
message.write_text('review(pr14): publish second-round real GPU FAST recheck\n\nPin formal production 4b5c480b separately from evidence delivery. Record\nreal Ns2/Ns10 chemistry, cold/restart, Co, clock, ALE, SST/RK, radiation\nand resource observations; retain wet SIGFPE and unresolved strict reds.\nAppend protocol rechecks and immutable evidence references. No production\nsource changes, PR conversation edits, merge or whole-PR pass claim.\n')
q=checked(['git','-C',str(ROOT),'commit','--file',str(message)]);(ROOT/'work/round2-commit.log').write_text(q.stdout+q.stderr)
commit=git('rev-parse','HEAD');print('COMMIT',commit,flush=True)
assert not git('diff','--name-only',before,commit,'--','applications','common','tests','tools','README.md')
assert set(git('diff','--name-only',before,commit).splitlines())==set(staged)
q=subprocess.run(['git','-C',str(ROOT),'push','origin','HEAD:refs/heads/'+branch],capture_output=True,text=True,env=env,timeout=300)
(ROOT/'work/round2-push.log').write_text(q.stdout+q.stderr);print('PUSH',q.returncode,q.stdout,q.stderr,flush=True)
result={'identity':'PR14_REVIEW_ONLY','commit':commit,'record_tree':git('rev-parse',commit+'^{tree}'),'pushed':q.returncode==0,'before':before,'branch':branch,'delivery':delivery,'production_source':SOURCE,'production_tree':PATHS['source_tree'],'staged_file_count':len(staged),'verified_SHA256_file_count':count,'no_production_diff':True,'merge_authorized':False,'utc':utc()}
(ROOT/'work/round2-publication-result.json').write_text(json.dumps(result,indent=2)+'\n')
assert q.returncode==0,'Normal push failed; preserve committed evidence and inspect remote changes before retry.'
