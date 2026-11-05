#!/usr/bin/env python3
"""Refresh/check literal diagnostic producers; shared sites name their first producer."""
import argparse
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];path=root/'common/DiagnosticsCatalogue.H'
    text=path.read_text();changes=[];ambiguous=[]
    original=text
    text=re.sub(r'(\{30900,)"[^"]+","[^"]+",',
        r'\1"applications/CHMT/materials/MaterialReactionMath.H","advanceMaterialReactionsMath",',text)
    pattern=r'(\{(\d+),"([^"]+)","[^"]+",)(\d+)(,)'
    def replace(m):
        source=root/m[3]
        if not source.exists():ambiguous.append((int(m[2]),m[3],'missing file'));return m[0]
        if int(m[2])==1:return m[0] # Runtime CUDA errors share recordCudaFailure.
        lines=[i+1 for i,line in enumerate(source.read_text().splitlines())
               if re.search(r'(?<![\w.])'+m[2]+r'(?![\w.])',line) and not line.lstrip().startswith('//') and '.sources.push_back' not in line]
        if not lines:ambiguous.append((int(m[2]),m[3],lines));return m[0]
        # Repeated literals describe branches of a shared site. Record the
        # first emitting statement, never a stale registration line.
        if lines[0]!=int(m[4]):changes.append((int(m[2]),int(m[4]),lines[0]))
        return m[1]+str(lines[0])+m[5]
    updated=re.sub(pattern,replace,text)
    for item in changes:print('line update',*item)
    for item in ambiguous:print('manual/computed site',*item)
    if args.check:
        if updated!=original or ambiguous:raise SystemExit(1)
    else:path.write_text(updated)

if __name__=='__main__':main()
