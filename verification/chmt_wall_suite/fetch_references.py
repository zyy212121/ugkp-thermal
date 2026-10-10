#!/usr/bin/env python3
"""Opt-in source fetcher; never vendors data into the code repository."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from foam import ROOT

def fetch(source_id,out,allow_noncommercial=False):
    out=Path(out).resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('Download external data outside the repository')
    sources=json.loads((Path(__file__).parent/'references/sources.json').read_text())
    source=next((s for s in sources if s['id']==source_id),None)
    if source is None:raise ValueError('unknown reference id')
    if source['license']=='CC-BY-NC-4.0' and not allow_noncommercial:raise ValueError('CC-BY-NC data: explicitly acknowledge noncommercial use with --allow-noncommercial-data; commercial users need separate permission')
    if out.exists():raise FileExistsError('refusing existing download '+str(out))
    with urllib.request.urlopen(source['url'],timeout=60) as response:data=response.read(64*1024*1024+1)
    if len(data)>64*1024*1024:raise ValueError('reference exceeds 64 MiB limit')
    if hashlib.sha256(data).hexdigest()!=source['sha256']:raise ValueError('reference hash changed; review source version before use')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    out.with_name(out.name+'.provenance.json').write_text(json.dumps(source,indent=2)+'\n')
    return source

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source_id');p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-noncommercial-data',action='store_true');a=p.parse_args()
    print(json.dumps(fetch(a.source_id,a.output,a.allow_noncommercial_data),indent=2))
if __name__=='__main__':main()
