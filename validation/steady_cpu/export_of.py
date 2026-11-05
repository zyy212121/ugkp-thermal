#!/usr/bin/env python3
"""Export actual ASCII OpenFOAM fields into comparable streamwise CSV.

Only fields present in the result are exported; absent diagnostics are listed,
never filled with synthetic expected values. Geometry follows reference-input.json.
"""
import argparse
import csv
import json
from pathlib import Path
import re

MAPPING={'rho':('rho',0),'rhoUy':('rhoU',1),'rhoUz':('rhoU',2),'rhoUx':('rhoU',0),'rhoE':('rhoE',0),'Uy':('U',1),'Uz':('U',2),'Ux':('U',0),'p':('p',0),'T':('T',0),'k':('k',0),'omega':('omega',0),'nut':('nut',0),'epsilonS':('epsilonS',0),'rhoUsx':('rhoUs',0),'rhoUsy':('rhoUs',1),'rhoUsz':('rhoUs',2),'rhoEs':('rhoEs',0),'rhoHp':('rhoHp',0),'rhoDs':('rhoDs',0),'Tp':('Tp',0),'theta':('theta',0)}

def internal(path,n):
    text=Path(path).read_text();text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    match=re.search(r'internalField\s+uniform\s+(\([^)]*\)|[^;]+);',text)
    if match: return [list(map(float,match[1].strip('() ').split()))]*n
    match=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not match: raise ValueError(f'No ASCII internalField in {path}')
    tokens=re.findall(r'\([^)]*\)|[^\s()]+',match[2]); rows=[list(map(float,t.strip('() ').split())) for t in tokens]
    if len(rows)!=n: raise ValueError(f'{path}: field has {len(rows)} rows, mesh manifest has {n}')
    return rows

def export(case,output,times=None):
    case=Path(case);manifest=json.loads((case/'reference-input.json').read_text());a=manifest['input'];n=int(a['cells']);dx=a['length']/n
    dirs=sorted((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?',p.name)),key=lambda p:float(p.name))
    if times is not None: dirs=[p for p in dirs if float(p.name) in times]
    rows=[];missing={}
    for directory in dirs:
        data={name:internal(directory/name,n) for name,_ in MAPPING.values() if (directory/name).exists()}
        for name in ['Y_A','Y_B','rhoY_A','rhoY_B']:
            if (directory/name).exists():data[name]=internal(directory/name,n)
        missing[directory.name]=sorted(set(name for name,_ in MAPPING.values())-set(data))
        for c in range(n):
            r={'iteration':int(float(directory.name)),'cell':c,'x':(c+.5)*dx,'volume':dx}
            r.update({column:data[name][c][component] for column,(name,component) in MAPPING.items() if name in data})
            if 'rho' in data:
                for j,name in enumerate(['Y_A','Y_B']):
                    if 'rho'+name in data:r['rhoY'+str(j)]=data['rho'+name][c][0]
                    elif name in data:r['rhoY'+str(j)]=data['rho'][c][0]*data[name][c][0]
                if 'epsilonS' in data:r['rhoSolid']=1000*data['epsilonS'][c][0]
            rows.append(r)
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with Path(output).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    report={'rows':len(rows),'missing_actual_fields':missing,'geometry_source':'reference-input.json','not_exported':['per-cell spatial residuals','pseudoDt','cellStatus','parcel count/age (not reconstructed from moments)']}
    Path(str(output)+'.manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report

def export_windows(case,output,times=None):
    rows=[]; sources=[]
    for path in sorted(Path(case).glob('*/steadyWindowMean.csv'),key=lambda p:float(p.parent.name)):
        iteration=int(float(path.parent.name))
        if times is not None and iteration not in times:continue
        with path.open() as f:
            reader=csv.DictReader(f)
            names=[name for name in reader.fieldnames if name!='cell']
            for source in reader:
                row={'iteration':iteration,'cell':source['cell']}
                row.update({'q'+str(j)+'Mean':source[name] for j,name in enumerate(names)})
                rows.append(row)
        sources.append({'path':str(path),'component_names':names})
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with Path(output).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    report={'rows':len(rows),'sources':sources,'scope':'Actual production time-weighted window means only; no reconstructed RMS or budgets.'}
    Path(str(output)+'.manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case');p.add_argument('output');p.add_argument('--times',type=float,nargs='+');p.add_argument('--windows',action='store_true');a=p.parse_args();print(json.dumps((export_windows if a.windows else export)(a.case,a.output,a.times),indent=2))
