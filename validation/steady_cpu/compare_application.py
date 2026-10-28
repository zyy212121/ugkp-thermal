#!/usr/bin/env python3
"""Compare the mathematical reference with exported real application fields.

Tolerance checks are evidence reports only, not simulation stopping rules.
Example: compare_application.py cpu.csv of.csv --fields rho rhoUx rhoE Ux p T
Use --particles for CPU .particles.csv versus actual moment fields, comparing
statistics rather than parcel ordering. Raw sampling-window budgets can be
retained with --log; no unobserved exchange is filled in or certified.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

PARTICLES={'momRhoP':'rhoSolid','momRhoUPx':'rhoUsx','momRhoUPy':'rhoUsy','momRhoUPz':'rhoUsz','momRhoEP':'rhoEs','momRhoHpP':'rhoHp','momRhoPD':'rhoDs'}

def read(path):
    with open(path) as f: rows=list(csv.DictReader(f))
    return rows

def comparison(reference,candidate,fields,rtol,atol,particles=False,start=None,end=None):
    left=read(reference);right=read(candidate);mapping=PARTICLES if particles else {}
    def index(rows):
        out={};dups=[]
        for r in rows:
            it=int(float(r['iteration']))
            if (start is not None and it<start) or (end is not None and it>end):continue
            key=(it,int(r['cell']))
            if key in out:dups.append(key)
            out[key]=r
        return out,dups
    left,ld=index(left);right,rd=index(right);keys=sorted(set(left)&set(right));out={'matching_rows':len(keys),'missing_candidate_keys':sorted(set(left)-set(right)),'extra_candidate_keys':sorted(set(right)-set(left)),'duplicate_reference_keys':ld,'duplicate_candidate_keys':rd,'fields':{},'unavailable':[]}
    for field in fields:
        other=mapping.get(field,field);stats={'max_absolute':0.,'max_scaled':0.,'mismatches':0,'samples':0}
        for key in keys:
            a,b=left[key].get(field),right[key].get(other)
            if a in (None,'') or b in (None,''):
                out['unavailable'].append({'key':key,'reference_field':field,'candidate_field':other});continue
            a,b=float(a),float(b);finite=math.isfinite(a) and math.isfinite(b);d=abs(a-b);limit=atol+rtol*max(abs(a),abs(b));scaled=d/limit if limit else (0 if d==0 else math.inf)
            stats['samples']+=1;stats['mismatches']+=int(not finite or d>limit)
            stats['max_absolute']=max(stats['max_absolute'],d) if finite else math.inf;stats['max_scaled']=max(stats['max_scaled'],scaled) if finite else math.inf
        out['fields'][field]=stats
    out['pass']=bool(keys and fields) and not any((out['missing_candidate_keys'],out['extra_candidate_keys'],ld,rd,out['unavailable'])) and all(v['samples'] and not v['mismatches'] for v in out['fields'].values())
    out['scope']='Only explicitly selected actual field values; no claim of CUDA validation or unexported budget closure.'
    return out

def budget_log(path):
    windows=[];current=None
    for line in Path(path).read_text().splitlines():
        if line.startswith('Steady sampling window '):
            current={'header':line,'reported_lines':[]};windows.append(current)
        elif current is not None and (line.startswith('  ') or 'species' in line.lower()):current['reported_lines'].append(line.strip())
    return {'windows':windows,'interpretation':'Verbatim actual application diagnostics. Intrinsic gas FV exchange is not a porosity-weighted total mixture budget; do not silently combine these quantities.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference');p.add_argument('candidate');p.add_argument('--fields',nargs='+');p.add_argument('--particles',action='store_true');p.add_argument('--rtol',type=float,default=1e-8);p.add_argument('--atol',type=float,default=1e-10);p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--log');a=p.parse_args()
    fields=a.fields or (list(PARTICLES) if a.particles else ['rho','rhoUx','rhoE','Ux','p','T'])
    result=comparison(a.reference,a.candidate,fields,a.rtol,a.atol,a.particles,a.start,a.end)
    if a.log:result['actual_sampling_budget']=budget_log(a.log)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
