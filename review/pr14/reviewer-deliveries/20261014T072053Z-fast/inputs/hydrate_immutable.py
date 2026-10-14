#!/usr/bin/env python3
"""PR14_REVIEW_ONLY. Restore only absent aliases whose immutable content hash is verified."""
from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parents[1]
ROOT=D.parents[3]
data=json.loads((D/'metadata/immutable-references.json').read_text())
for q in data['references']:
 p=D/q['alias']
 assert p.resolve().is_relative_to(D.resolve()), q['alias']
 content=subprocess.check_output(['git','-C',str(ROOT),'show',q['fixed_commit']+':'+q['repo_path']])
 assert hashlib.sha256(content).hexdigest()==q['sha256'], q['alias']
 if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==q['sha256'], q['alias']
 else: p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
print('Verified/restored aliases:',len(data['references']))
