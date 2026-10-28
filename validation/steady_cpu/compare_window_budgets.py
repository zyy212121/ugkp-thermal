#!/usr/bin/env python3
"""Compare CPU per-cell sampling inventories/RMS with actual application log.

Windows are matched in order; use identical CFG outputEvery/windowSamples,
physical dt and run length. This compares measured inventory endpoints and
statistical reports, NOT a fabricated total coupled boundary-energy closure.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re

BASE=['rho','rhoUx','rhoUy','rhoUz','rhoE','rhoK','rhoOmega','rhoSolid','rhoUsx','rhoUsy','rhoUsz','rhoEs','rhoHp','rhoDs']

def compare(reference,log,rtol=1e-5,atol=1e-9):
    grouped={}
    with open(reference) as f:
        for row in csv.DictReader(f):grouped.setdefault(int(row['iteration']),[]).append(row)
    actual=[]
    for line in Path(log).read_text().splitlines():
        if line.startswith('Steady sampling window '):actual.append({'header':line,'components':{},'exchanges':[]})
        elif actual and 'inventoryBegin=' in line:
            words=line.split();actual[-1]['components'][words[0]]={k:float(v) for k,v in (token.split('=',1) for token in words[1:])}
        elif actual and line.startswith('  ') and ':' in line:actual[-1]['exchanges'].append(line.strip())
    result={'reference_windows':len(grouped),'actual_windows':len(actual),'windows':[],'pass':bool(grouped) and len(grouped)==len(actual),'scope':'Measured endpoints and window statistics only; original application exchange reports retained. Not total mixture energy closure.'}
    for (iteration,rows),window in zip(sorted(grouped.items()),actual):
        count=len([name for name in rows[0] if re.fullmatch(r'q\d+Mean',name)])
        result['pass'] &= count>0 and len({row['cell'] for row in rows})==len(rows)
        report={'iteration':iteration,'actual_header':window['header'],'components':{},'actual_exchanges':window['exchanges']}
        for q in range(count):
            name=BASE[q] if q<14 else 'rhoY'+str(q-14);entry={}
            for source,target,aggregate in [('MeanChange','meanChange',max),('Fluctuation','relativeRms',max),('InventoryBegin','inventoryBegin',sum),('InventoryEnd','inventoryEnd',sum)]:
                expected=aggregate(float(row[f'q{q}{source}']) for row in rows);observed=window['components'].get(name,{}).get(target)
                error=None if observed is None else abs(expected-observed)
                passed=observed is not None and math.isfinite(expected) and math.isfinite(observed) and error<=atol+rtol*max(abs(expected),abs(observed))
                entry[target]={'reference':expected,'actual':observed,'absolute_error':error,'pass':passed};result['pass'] &= passed
            report['components'][name]=entry
        result['windows'].append(report)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference_windows');p.add_argument('application_log');p.add_argument('--rtol',type=float,default=1e-5,help='Default allows default OpenFOAM stdout precision; use tighter tolerance with higher log precision.');p.add_argument('--atol',type=float,default=1e-9);a=p.parse_args();report=compare(a.reference_windows,a.application_log,a.rtol,a.atol);print(json.dumps(report,indent=2));raise SystemExit(0 if report['pass'] else 1)
