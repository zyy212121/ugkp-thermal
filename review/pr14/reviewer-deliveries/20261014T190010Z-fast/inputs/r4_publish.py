from r4_runner import *
branch='perf/chmt-first-round-20261013';review=ROOT/'review/pr14';delivery=str(D.relative_to(ROOT));env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
def check(*argv):
 q=subprocess.run(list(map(str,argv)),capture_output=True,text=True,env=env);assert q.returncode==0,(argv,q.returncode,q.stdout[-2000:],q.stderr[-2000:]);return q
assert not git('diff','--cached','--name-only');assert git('rev-parse','HEAD')==PATHS['delivery_parent']
check('git','-C',ROOT,'fetch','origin',branch);before=git('rev-parse','origin/'+branch);assert before==PATHS['delivery_parent'],('Remotechanged; preserve records andintegrate before normalpush',before)
assert not git('diff','--name-only','HEAD','--','applications','common','tests','test','tools','README.md')
count=0
for line in (D/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert sha(D/n)==h,n;count+=1
assert all(p.stat().st_size<100*1024*1024 for p in D.rglob('*') if p.is_file()),'GitHub100MB file limit; retain losslesssplit evidence'
validator=check('python3',review/'validate_records.py');assert json.loads(validator.stdout)['errors']==[]
check('git','-C',ROOT,'add','-f','--',delivery);check('git','-C',ROOT,'add','--','review/pr14/manifest.json','review/pr14/INDEX.md','review/pr14/pending-findings.json','review/pr14/responses')
staged=git('diff','--cached','--name-only').splitlines();allowed=[delivery,'review/pr14/manifest.json','review/pr14/INDEX.md','review/pr14/pending-findings.json','review/pr14/responses']
assert all(any(p==a or p.startswith(a+'/') for a in allowed) for p in staged);assert not any(Path(p).name.lower().startswith('readme') for p in staged)
for p in D.rglob('*'):
 if p.is_file():assert hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT),'show',':'+str(p.relative_to(ROOT))])).hexdigest()==sha(p),p
check('git','-C',ROOT,'diff','--cached','--check','--','review/pr14/INDEX.md','review/pr14/manifest.json','review/pr14/pending-findings.json','review/pr14/responses',delivery+'/REPORT.md',delivery+'/coverage.json',delivery+'/findings.json')
message=ROOT/'work/r4-commit-message.txt';message.write_text('review(pr14): publish FAST4 independent real GPU evidence\n\nPin production47034e4e separately from records. Fresh original native\nflags builds, chemical retry/terminal controls and supported negative\ntransactions, Ns10 full700-step chemical half volumes, actual low-Re\nSST Euler/RK/gross scopes, complete original evaporation endpoint,\nstrict restart failures, actual resource retry and profile data.\nRetain all14 partial scope gaps and retired support metric.\nPreserve formal fixture/API red; no production change or merge.\n')
q=check('git','-C',ROOT,'commit','--file',message);(ROOT/'work/r4-commit.log').write_text(q.stdout+q.stderr);commit=git('rev-parse','HEAD');print('COMMIT',commit,len(staged),'files',flush=True)
assert not git('diff','--name-only',before,commit,'--','applications','common','tests','test','tools','README.md')
pushlog=ROOT/'work/r4-push.log'
with pushlog.open('w') as h:proc=subprocess.Popen(['git','-C',str(ROOT),'push','--progress','origin','HEAD:refs/heads/'+branch],stdout=h,stderr=subprocess.STDOUT,env=env)
while True:
 try:code=proc.wait(timeout=30);break
 except subprocess.TimeoutExpired:print('PUSH_PROGRESS',pushlog.read_text(errors='replace')[-700:],flush=True)
print('PUSH',code,pushlog.read_text(errors='replace')[-2000:],flush=True)
receipt={'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','review_round':'FAST4','primary_record_commit':commit,'primary_record_tree':git('rev-parse',commit+'^{tree}'),'pushed':code==0,'remote_before':before,'branch':branch,'delivery':delivery,'production_source':SOURCE,'production_tree':PATHS['source_tree'],'new_file_count':len(staged),'verified_SHA256_payload_count':count,'no_production_or_README_diff':True,'whole_FAST_acceptance':False,'merge_ready':False,'merge_authorized':False,'utc':utc()};(ROOT/'work/r4-publication-result.json').write_text(json.dumps(receipt,indent=2)+'\n');assert code==0,'Normalpushfailed; inspect/integrate remote, noforce.'
