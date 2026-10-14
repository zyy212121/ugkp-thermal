import json,subprocess,os,collections
from r2_runner import *
a=json.loads((D/'results.json').read_text());ids={x['id'] for x in a['tests']}
bad=[]
for x in a['tests']:
 p=D/x['log']
 if not p.is_file() or sha(p)!=x['log_sha256']:bad.append(x['id'])
orphans=[p.name for p in (D/'logs').glob('*.spec.json') if p.name[:-10] not in ids]
record={'utc':utc(),'event':'User reports workstation crash and requests continuation','result_count':len(ids),'log_hash_mismatches':bad,'orphan_specs':orphans,'processes':subprocess.check_output(['ps','-eo','pid,etime,args'],text=True),'gpu':subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True),'production_diff':git('diff','--stat','--','applications','common','tests'),'continuation_policy':'Reuse completed immutable results; serial GPU workloads; crash cause unknown.'}
(D/'metadata/crash-resume.json').write_text(json.dumps(record,indent=2)+'\n')
print('Integrity',len(ids),bad,orphans,'production_diff',record['production_diff'])
print('Verdicts',collections.Counter(x['verdict'] for x in a['tests']))
for x in a['tests']:
 if x['verdict']!='PASS': print(x['id'],x['verdict'],x['exit_code'],x['observations'][:350])
for rel in ['logs/ACTUAL-MOVING-CHEM-VOLUMES.log','artifacts/extra/moving-chemistry/actual-chemical-kernel-volume-inputs.json','artifacts/extra/moving-chemistry/volume-observer-output-comparison.json']:
 p=D/rel
 print(rel, p.read_text(errors='replace')[-8000:] if p.exists() else 'missing')
print('fixtures',[(p.name,p.stat().st_size) for p in (ROOT/'work/fixtures').glob('*') if p.is_file()])
