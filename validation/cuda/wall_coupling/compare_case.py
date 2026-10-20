#!/usr/bin/env python3
"""Compare a GPU OpenFOAM output with an already delivered CPU binary snapshot."""
from pathlib import Path
import argparse, re, csv, json
import numpy as np
def field(p,vec=False):
    b=p.read_bytes();kind=b'vector' if vec else b'scalar'
    m=re.search(rb'internalField\s+nonuniform\s+List<'+kind+rb'>\s+(\d+)\s*\(',b)
    if m:
        n=int(m[1])
        if re.search(rb'format\s+binary;',b[:1000]):
            a=np.frombuffer(b,dtype='<f8',count=n*(3 if vec else 1),offset=m.end()).copy()
        else:
            end=b.index(b';',m.end());a=np.fromstring(b[m.end():end].replace(b'(',b' ').replace(b')',b' ').decode(),sep=' ')
        return a.reshape((-1,3)) if vec else a
    m=re.search(rb'internalField\s+uniform\s+([^;]+);',b);assert m,p
    a=np.fromstring(m[1].replace(b'(',b' ').replace(b')',b' ').decode(),sep=' ')
    return np.tile(a,(26880,1)) if vec else np.full(26880,a[0])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case',type=Path,required=True); ap.add_argument('--cpu',type=Path,required=True); ap.add_argument('--audit',type=Path); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    t=max((p for p in a.case.iterdir() if p.is_dir() and re.fullmatch('[0-9eE.+-]+',p.name)),key=lambda p:float(p.name))
    data={k:field(t/k,k=='U') for k in ['U','T','p','rho','rhoE','k','omega']}; out={'time':float(t.name),'fields':{}}
    for k,v in data.items():
        v=v[:,0] if k=='U' else v; q=np.fromfile(a.cpu/({'U':'Ux','T':'Tgas'}.get(k,k)+'.bin')); e=abs(v-q)
        out['fields'][k]={'max_abs':float(e.max()),'max_relative':float((e/np.maximum(1.,abs(q))).max()),'worst_cell':int(e.argmax())}
    walls=list(csv.DictReader((t/'uniform/gasBoundaryLayer.csv').open()))
    out['reference_final']={'max_residual':max(float(x['residual']) for x in walls),'failed':sum(float(x['residual'])>1.01e-8 for x in walls)}
    work=[]
    if a.audit:
        work=list(csv.DictReader((a.audit/'wall-work.csv').open()))
        out['work']={k:sum(int(x[k]) for x in work) for k in ['solves','failures','iterations','residuals','jacobians','central','history','fallbacks']}
        out['work']['max_residual']=max(float(x['maxResidual']) for x in work)
        out['work']['max_moment_residual']=max(float(x['momentResidual']) for x in work)
        out['work']['active']=sum(int(x['active']) for x in work)
    byowner={int(x['owner']):x for x in work}; out['stations']=[]
    for c in [90,7684,7782,7864,23080]:
        row={'cell':c,**{k:float(v[c,0] if k=='U' else v[c]) for k,v in data.items()}}
        if c in byowner:row.update({k:float(byowner[c][k]) for k in ['tractionX','ownerSourceK','donorSourceK','donorSourceOmega']})
        out['stations'].append(row)
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    if any(not np.isfinite(v).all() for v in data.values()):raise SystemExit(2)
if __name__=='__main__':main()
