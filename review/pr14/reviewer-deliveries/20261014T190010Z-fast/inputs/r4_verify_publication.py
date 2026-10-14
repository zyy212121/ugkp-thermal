from r4_runner import *
import urllib.request
record=ROOT/'work/r4-publication-result.json';v=json.loads(record.read_text());assert v['pushed'];commit=v['primary_record_commit'];branch=v['branch'];review=ROOT/'review/pr14';env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never',GIT_LFS_SKIP_SMUDGE='1')
remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin','refs/heads/'+branch],text=True,env=env).split()[0];assert remote==commit,(remote,commit)
def request(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PR14-independent-FAST4-review'}),timeout=30) as r:return r.read()
pr=json.loads(request('https://api.github.com/repos/zyy212121/ugkp-thermal/pulls/14'));assert pr['head']['sha']==commit and pr['head']['ref']==branch and pr['state']=='open' and not pr['merged']
raw=request('https://raw.githubusercontent.com/zyy212121/ugkp-thermal/'+commit+'/'+str((D/'REPORT.md').relative_to(ROOT)));assert hashlib.sha256(raw).hexdigest()==sha(D/'REPORT.md')
source=json.loads((D/'metadata/source-hashes.json').read_text());assert all(git('rev-parse',commit+':'+q['path'])==q['blob'] for q in source)
checks=0
for line in (D/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert sha(D/n)==h,n;checks+=1
q=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);assert q.returncode==0 and json.loads(q.stdout)['errors']==[]
assert not git('diff','--cached','--name-only') and not git('diff','--name-only','--','review/pr14')
v.update(verified_utc=utc(),verified_remote_primary_head=remote,verified_GitHub_PR_primary_head=pr['head']['sha'],PR_open=True,PR_merged=False,raw_report_sha256=hashlib.sha256(raw).hexdigest(),source_blobs_verified=len(source),SHA256_payloads_verified=checks,document_validation=json.loads(q.stdout),verification_scope='Actualnormalpushprimaryhead+GitHubPRopen/unmerged+rawfixedcommitreportSHA+1264productionblobs+allnewSHAfiles+manifestvalidatorerrors[]; CI notrerun/claimed; noPRcomments/description/merge',publication_receipt_scope='This receipt verifies the primary record commit above. A later record-only metadata commit publishes this receipt; no self-referential containingcommit hash is fabricated.')
record.write_text(json.dumps(v,indent=2)+'\n');(D/'PUBLICATION.json').write_text(json.dumps(v,indent=2)+'\n');shutil.copy2(Path(__file__),D/'inputs/r4_verify_publication.py')
# Only append receipt evidence/ownership; preserve all immutable report/source/result payloads.
m=json.loads((review/'manifest.json').read_text());added=[]
for p in [D/'PUBLICATION.json',D/'inputs/r4_verify_publication.py']:
 h=sha(p);eid='E014-'+h[:16];assert not any(q['id']==eid for q in m['evidence']);m['evidence'].append({'id':eid,'path':str(p.relative_to(review)),'sha256':h,'bytes':p.stat().st_size,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'scope':'Actualremoteverification of immutableFAST4primaryrecord, separate from production andreceiptcontainingcommit.'});added.append(eid)
delivery=next(q for q in m['reviewer_deliveries'] if q['batch']==D.name);delivery['publication_commit']=commit;delivery['remote_primary_verified']=True;delivery['publication_receipt']=str((D/'PUBLICATION.json').relative_to(review));delivery['evidence_ids']+=added
for t in m['topics']:
 t['evidence_ids']=list(dict.fromkeys(t['evidence_ids']+added));p=review/t['response_path'];r=json.loads(p.read_text());r.setdefault('reviewer_publication_verifications',[]).append({'id':t['id']+'-PUBLISHED-'+D.name,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','recorded_utc':utc(),'source_commit':SOURCE,'source_tree':PATHS['source_tree'],'primary_record_commit':commit,'verified_remote_head_at_primary_verification':remote,'report_sha256':v['raw_report_sha256'],'evidence_ids':added,'whole_FAST_acceptance':False,'merge_authorized':False});p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
log=D/'logs/validate-publication-records.json';log.write_text('{}\n');m['owned_paths']=sorted('review/pr14/'+str(p.relative_to(review)) for p in review.rglob('*') if p.is_file());(review/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
q=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);log.write_text(q.stdout+q.stderr);assert q.returncode==0
with (D/'SHA256SUMS').open('w') as h:
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS':h.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
expected=[str((D/'PUBLICATION.json').relative_to(ROOT)),str((D/'inputs/r4_verify_publication.py').relative_to(ROOT)),str(log.relative_to(ROOT)),str((D/'SHA256SUMS').relative_to(ROOT)),'review/pr14/manifest.json']+[t['response_path'] if t['response_path'].startswith('review/pr14/') else 'review/pr14/'+t['response_path'] for t in m['topics']]
subprocess.run(['git','-C',str(ROOT),'add','-f','--',*expected],check=True,env=env);assert set(git('diff','--cached','--name-only').splitlines())==set(expected)
message=ROOT/'work/r4-publication-commit-message.txt';message.write_text('review(pr14): record verified FAST4 remote publication\n\nVerify primary evidence commit '+commit+' against GitHub PR head,\nraw report SHA,1264 production blobs and all evidence checksums.\nAppend protocol receipt separately from its own containing commit.\nNo production change, tolerance edit, PR comment or merge.\n');subprocess.run(['git','-C',str(ROOT),'commit','--file',str(message)],check=True,env=env);head=git('rev-parse','HEAD');assert not git('diff','--name-only',commit,head,'--','applications','common','tests','test','tools','README.md')
fresh=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin','refs/heads/'+branch],text=True,env=env).split()[0];assert fresh==commit,'Remoteadvanced; integrate changes, noforce'
subprocess.run(['git','-C',str(ROOT),'push','origin','HEAD:refs/heads/'+branch],check=True,env=env)
remote=subprocess.check_output(['git','-C',str(ROOT),'ls-remote','origin','refs/heads/'+branch],text=True,env=env).split()[0];assert remote==head;pr=json.loads(request('https://api.github.com/repos/zyy212121/ugkp-thermal/pulls/14'));assert pr['head']['sha']==head and not pr['merged'] and pr['state']=='open'
published=request('https://raw.githubusercontent.com/zyy212121/ugkp-thermal/'+head+'/'+str((D/'PUBLICATION.json').relative_to(ROOT)));assert hashlib.sha256(published).hexdigest()==sha(D/'PUBLICATION.json')
q=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);assert q.returncode==0;assert not git('diff','--name-only','--','review/pr14') and not git('diff','--cached','--name-only')
v.update(final_remote_record_commit=head,final_record_tree=git('rev-parse',head+'^{tree}'),final_PR_head=pr['head']['sha'],final_verified_utc=utc(),remote_publication_receipt_sha256=hashlib.sha256(published).hexdigest(),final_document_validation=json.loads(q.stdout));record.write_text(json.dumps(v,indent=2)+'\n');shutil.copy2(record,P/'r4-publication-result.json');shutil.copy2(D/'REPORT.md',P/'PR14-FAST4-REPORT.md');shutil.copy2(D/'coverage.json',P/'PR14-FAST4-coverage.json');print(json.dumps(v,indent=2),flush=True)
