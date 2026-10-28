#!/usr/bin/env python3
"""Compare matching CPU/CUDA CSV fields, without parcel-order assumptions.

Use matching cell/iteration keys for field files. Particle comparison must use
cell statistics (mass, momentum, energy, counts), not atomic insertion order.
This tool reports discrepancies; it does not certify unsupported physics.
"""
import argparse
import csv
import json
import math

def compare(reference, candidate, rtol=1e-8, atol=1e-10):
    def read(path):
        with open(path) as stream:
            rows=list(csv.DictReader(stream))
        indexed={};duplicates=[]
        for r in rows:
            key=(r['iteration'],r['cell'])
            if key in indexed: duplicates.append(key)
            indexed[key]=r
        return indexed,duplicates
    (left,ld),(right,rd)=read(reference),read(candidate)
    result={'missing_keys':sorted(set(left)-set(right)), 'extra_keys':sorted(set(right)-set(left)), 'fields':{},'duplicate_reference_keys':ld,'duplicate_candidate_keys':rd,'comparable_rows':len(set(left)&set(right)), 'schema_discrepancies':[]}
    for key in sorted(set(left)&set(right)):
        if set(left[key])!=set(right[key]): result['schema_discrepancies'].append(key)
        for field in (set(left[key])&set(right[key]))-{'iteration','cell'}:
            a,b=float(left[key][field]),float(right[key][field])
            d=abs(a-b);limit=atol+rtol*max(abs(a),abs(b))
            entry=result['fields'].setdefault(field,{'max_absolute':0.,'max_scaled':0.,'mismatches':0})
            entry['max_absolute']=max(entry['max_absolute'],d)
            entry['max_scaled']=max(entry['max_scaled'],d/limit if limit else (0 if d==0 else math.inf))
            entry['mismatches']+=int(not(math.isfinite(a) and math.isfinite(b)) or d>limit)
    result['missing_fields']=sorted(set(next(iter(left.values()),{}))-set(next(iter(right.values()),{})))
    result['pass']=not(not result['comparable_rows'] or not result['fields'] or ld or rd or result['schema_discrepancies'] or result['missing_keys'] or result['extra_keys'] or result['missing_fields'] or any(e['mismatches'] for e in result['fields'].values()))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference');p.add_argument('candidate');p.add_argument('--rtol',type=float,default=1e-8);p.add_argument('--atol',type=float,default=1e-10)
    a=p.parse_args();result=compare(a.reference,a.candidate,a.rtol,a.atol);print(json.dumps(result,indent=2));raise SystemExit(0 if result['pass'] else 1)
