#!/usr/bin/env python3
"""PR14_REVIEW_ONLY: standalone, read-only documentation/evidence consistency check."""
from pathlib import Path
import json,hashlib
D=Path(__file__).resolve().parent
m=json.loads((D/'manifest.json').read_text())
assert m['identity']=='PR14_REVIEW_ONLY'
errors=[]
owned=set(m['owned_paths']);actual={'review/pr14/'+str(p.relative_to(D)) for p in D.rglob('*') if p.is_file()}
if owned!=actual:errors.append({'ownership_difference':sorted(owned^actual)})
eids=set()
for e in m['evidence']:
 p=D/e['path'];eids.add(e['id'])
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:errors.append({'evidence_hash':e['id']})
for t in m['topics']:
 for field in ['path','response_path']:
  if not (D/t[field]).is_file():errors.append({'missing':t[field]})
 for eid in t['evidence_ids']:
  if eid not in eids:errors.append({'missing_evidence':eid})
 r=json.loads((D/t['response_path']).read_text())
 if r.get('topic_id')!=t['id'] or r.get('identity')!='PR14_REVIEW_ONLY':errors.append({'response_identity':t['id']})
 for f in t['file_ids']:
  if not (D/'files'/f'{f}.md').is_file():errors.append({'missing_file_card':f})
source=json.loads((D/'source-index.json').read_text())
print(json.dumps({'identity':'PR14_REVIEW_ONLY','administrative_only':True,'evidence_files':len(eids),'topics':len(m['topics']),'source_paths':len(source['files']),'errors':errors,'gpu_or_numerical_tests_run':False},indent=2))
raise SystemExit(bool(errors))
