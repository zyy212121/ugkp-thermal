#!/usr/bin/env python3
"""Validate a saved correctness bundle. GPU execution requires explicit --run.

No performance measurement. Every binary/source hash is checked before running.
The bundle is produced with the offline verification record, not from Git's
working-tree index (which may contain intentional historical case removals).
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--run',action='store_true',help='Execute CUDA correctness programs after hash validation')
a=p.parse_args();bundle=a.bundle.resolve();manifest=json.loads((bundle/'deferred-run.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,h in manifest['source_sha256'].items():
    path=(ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha(path)!=h:sys.exit('Source changed; rebuild the bundle before GPU execution: '+name)
for row in manifest['programs']:
    path=(bundle/row['executable']).resolve()
    if not path.is_relative_to(bundle) or not path.is_file() or sha(path)!=row['sha256']:sys.exit('Binary changed/missing: '+row['executable'])
print('Validated',len(manifest['programs']),'programs and',len(manifest['source_sha256']),'source hashes',flush=True)
if not a.run:print('Validation only. GPU execution remains pending; use --run during a correctness window.');sys.exit(0)
logs=bundle/'gpu-correctness';logs.mkdir(exist_ok=True);results=[]
for row in manifest['programs']:
    for index,args in enumerate(row['arguments']):
        q=subprocess.run([str(bundle/row['executable']),*args],capture_output=True,text=True,timeout=600)
        label=row['test']+'-'+str(index);(logs/(label+'.log')).write_text(q.stdout+q.stderr)
        ok=q.returncode==0 and row['expected_stdout'] in q.stdout
        results.append(dict(test=row['test'],arguments=args,exit=q.returncode,passed=ok))
        (logs/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(label,'PASS' if ok else 'FAIL',flush=True)
        if not ok:print(q.stdout+q.stderr);sys.exit(1)
print('GPU correctness complete:',len(results),'checks. No performance timing.',flush=True)
