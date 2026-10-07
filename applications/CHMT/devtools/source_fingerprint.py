#!/usr/bin/env python3
"""Print a deterministic CHMT source manifest without touching any source file."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

UPSTREAM_BASE='db455604156419a9e20b13f1b43694fde33be6c8'
EXCLUDED={'lnInclude','.build','__pycache__','.git','linux64GccDPInt32Opt','linux64GccDPInt64Opt'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def encoded(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()


def manifest(app):
    app=Path(app).resolve(); root=app.parents[1]
    files={str(p.relative_to(root)):sha(p) for p in sorted(app.rglob('*'))
           if p.is_file() and not any(x in EXCLUDED for x in p.relative_to(app).parts)
           and not (p.relative_to(app).parts[0]=='Make' and len(p.relative_to(app).parts)>2)
           and not p.name.endswith(('.pyc','.o','.so','.a','.dep'))}
    common={}; pending=[root/name for name in files]
    seen=set()
    while pending:
        path=pending.pop()
        if path in seen: continue
        seen.add(path)
        if path.suffix not in ('.H','.h','.C','.c','.cpp','.cu','.cuh'): continue
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"',path.read_text(),re.M):
            candidates=[path.parent/include,app/include,root/include,root/'common'/include]
            resolved=next((p.resolve() for p in candidates if p.is_file()),None)
            if resolved is not None and (root/'common') in resolved.parents:
                common[str(resolved.relative_to(root))]=sha(resolved); pending.append(resolved)
    git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],text=True,capture_output=True)
    if git.returncode: raise ValueError('build requires the actual git checkout revision')
    status=subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=all','--','applications/CHMT','common'],text=True,capture_output=True,check=True)
    dirty_paths=[line for line in status.stdout.splitlines() if not any(part in EXCLUDED for part in Path(line[3:]).parts)]
    identity={'upstream_base':UPSTREAM_BASE,'chmt_sources':files,'direct_and_transitive_common_sources':dict(sorted(common.items()))}
    return {**identity,'source_fingerprint':hashlib.sha256(encoded(identity)).hexdigest(),
            'solver_commit':git.stdout.strip(),'dirty':bool(dirty_paths),'dirty_paths':dirty_paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--app',type=Path,default=Path(__file__).resolve().parents[1]); args=parser.parse_args()
    print(json.dumps(manifest(args.app),indent=2,sort_keys=True))
if __name__=='__main__': main()
