#!/usr/bin/env python3
"""Summarise exported windows; fluctuations are reported, never solver gates."""
import argparse
import csv
import json
import statistics


def summarize(path,start=0):
    with open(path) as f: rows=[r for r in csv.DictReader(f) if int(r['iteration'])>=start]
    cells={}
    for r in rows: cells.setdefault(r['cell'],[]).append(r)
    output={}
    for cell,items in sorted(cells.items(),key=lambda p:int(p[0])):
        output[cell]={}
        for field in items[0].keys()-{'iteration','cell'}:
            values=[float(r[field]) for r in items]
            output[cell][field]={'mean':statistics.fmean(values),'rms_fluctuation':statistics.pstdev(values),'first':values[0],'last':values[-1],'samples':len(values)}
    return {'start_iteration':start,'rows':len(rows),'cells':output,
            'note':'Uniform-output sample means, not production dt-weighted every-step windows; do not claim identical convergence statistics.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv');p.add_argument('--start',type=int,default=0);a=p.parse_args()
    print(json.dumps(summarize(a.csv,a.start),indent=2,sort_keys=True))
