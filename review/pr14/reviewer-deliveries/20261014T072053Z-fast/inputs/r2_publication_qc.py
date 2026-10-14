from r2_runner import *
from collections import Counter
review=ROOT/'review/pr14';m=json.loads((review/'manifest.json').read_text());r=json.loads((D/'results.json').read_text());c=json.loads((D/'coverage.json').read_text());f=json.loads((D/'findings.json').read_text());refs=json.loads((D/'metadata/immutable-references.json').read_text())['references'];refmap={q['alias']:q for q in refs};testmap={q['id']:q for q in r['tests']}
assert len(testmap)==len(r['tests'])==221
assert all(sha(D/q['log'])==q['log_sha256'] for q in r['tests'])
assert all((D/q['log'].replace('.log','.spec.json')).is_file() for q in r['tests'])
for q in r['acceptance_units']:
 assert all(t in testmap for t in q['test_ids']),q['id']
 if q['verdict']=='PASS':assert q['test_ids'],q['id']
for q in f['findings']:
 for name in q.get('evidence',[]):assert (D/name).is_file() or name in refmap,name
units={q['id'] for q in r['acceptance_units']}
for q in c['topics']:assert set(q['acceptance_unit_ids']).issubset(units),q['topic_id']
source=json.loads((D/'metadata/source-hashes.json').read_text());assert len(source)==1152
assert all(sha(ROOT/q['path'])==q['sha256'] for q in source)
assert not git('diff','--name-only','HEAD','--','applications','common','tests','tools','README.md')
for name in ['REPORT.md','TEST_PLAN.md','coverage.json','results.json','findings.json','commands.txt','SOURCE.json','environment.json','versions.json','REPRODUCE.md','SHA256SUMS']:assert (D/name).is_file()
invalid=[]
for p in D.rglob('*'):
 if not p.is_file():continue
 if p.suffix=='.pyc' or p.name in ['id_rsa','id_ed25519'] or p.stat().st_size>90*1024*1024:invalid.append(str(p.relative_to(D)))
assert not invalid,invalid
status={'identity':'PR14_REVIEW_ONLY','verified_utc':utc(),'source_commit':SOURCE,'source_tree':PATHS['source_tree'],'source_file_count':len(source),'all_source_hashes_match':True,'raw_log_and_spec_count':len(r['tests']),'all_raw_log_hashes_match':True,'acceptance_unit_count':len(r['acceptance_units']),'acceptance_counts':dict(Counter(q['verdict'] for q in r['acceptance_units'])),'all_unit_test_refs_resolve':True,'all_finding_evidence_refs_resolve':True,'immutable_aliases':len(refs),'immutable_Git_blobs_verified_by_finalize':True,'production_diff_empty':True,'note':'Administrative evidence validation only; no new GPU executions and no whole-PR pass claim.'}
(D/'metadata/publication-integrity.json').write_text(json.dumps(status,indent=2)+'\n')
# This is the final administrative source version; the earlier unused push driver
# remains explicitly retained rather than mutating its already registered bytes.
copied=D/'inputs/r2_commit_push.py';previous=D/'inputs/r2_commit_push-pre-QC.py'
if sha(copied)!=sha(P/'r2_commit_push.py'):
 shutil.copy2(copied,previous)
 olddigest=sha(copied)
 for e in m['evidence']:
  if e['path']==str(copied.relative_to(review)) and e['sha256']==olddigest:e['path']=str(previous.relative_to(review))
 shutil.copy2(P/'r2_commit_push.py',copied)
shutil.copy2(P/'r2_publication_qc.py',D/'inputs/r2_publication_qc.py')
scripts=[{'path':str(p.relative_to(D)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((D/'inputs').glob('r2_*')) if p.is_file()]
drivers=D/'metadata/driver-scripts.json';original=drivers.read_bytes();backup=D/'metadata/driver-scripts-pre-QC.json';backup.write_bytes(original)
for e in m['evidence']:
 if e['path']==str(drivers.relative_to(review)) and e['sha256']==hashlib.sha256(original).hexdigest():e['path']=str(backup.relative_to(review))
drivers.write_text(json.dumps(scripts,indent=2)+'\n')
existing={e['sha256']:e['id'] for e in m['evidence']}
for p in [D/'metadata/publication-integrity.json',copied,previous,D/'inputs/r2_publication_qc.py',drivers,backup]:
 digest=sha(p)
 if digest not in existing:
  e={'id':'E014-'+digest[:16],'path':str(p.relative_to(review)),'sha256':digest,'bytes':p.stat().st_size,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'scope':'Administrative final integrity and publication driver; no numeric acceptance.'}
  m['evidence'].append(e);existing[digest]=e['id']
m['owned_paths']=sorted('review/pr14/'+str(p.relative_to(review)) for p in review.rglob('*') if p.is_file())
(review/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
v=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);assert v.returncode==0,v.stdout+v.stderr;(D/'logs/validate-records.json').write_text(v.stdout+v.stderr)
with (D/'SHA256SUMS').open('w') as h:
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS':h.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
print(json.dumps(status,indent=2));print(v.stdout,flush=True)
