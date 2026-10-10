#!/usr/bin/env python3
"""Import checksum-pinned numeric references outside the repository, no fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from foam import ROOT


def numeric_rows(text):
    rows=[];headers=[]
    for line in text.splitlines():
        if not line.strip():continue
        try:row=[float(x.replace('D','e').replace('d','e')) for x in line.split()]
        except ValueError:headers.append(line);continue
        if not row or not all(math.isfinite(x) for x in row):raise ValueError('nonfinite reference data')
        rows.append(row)
    return rows,headers

def import_source(source_id,path,output,acknowledge_nc=False):
    sources=json.loads((Path(__file__).parent/'references/sources.json').read_text());source=next((s for s in sources if s['id']==source_id),None)
    if source is None:raise ValueError('unknown source')
    output=Path(output).resolve()
    if output.is_relative_to(ROOT.resolve()):raise ValueError('external reference data must remain outside this repository')
    if output.exists():raise FileExistsError('refusing existing reference output')
    if source.get('license')=='CC-BY-NC-4.0' and not acknowledge_nc:raise ValueError('explicit --acknowledge-noncommercial-license is required; commercial use needs separate permission')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=source['sha256']:raise ValueError('source SHA256 mismatch; do not relabel altered data')
    rows,headers=numeric_rows(raw.decode('utf-8-sig'));warnings=[];axes={}
    if source_id.startswith('nasa_'):
        if not rows or any(len(r)!=2 for r in rows):raise ValueError('NASA expected two numeric columns')
        if 'upyp' in source_id:
            names=['log10_yplus','uplus'];warnings=['This profile is CFL3D SST numerical reference near Re_theta=10000, not direct measurements.']
        elif 'sst_cf' in source_id:names=['Cf','Re_theta'];warnings=['CFL3D SST numerical reference; requires matching turbulence model and operating conditions.']
        elif 'ks_cf' in source_id:names=['Re_theta','Cf'];warnings=['Karman-Schoenherr correlation, not raw experimental data; verify source column order from preserved header before comparison.']
        else:raise ValueError('NASA source columns not declared')
    elif source_id=='ablantis_exp2_tc.dat':
        names=headers[0].split()
        if any(len(r)!=len(names) for r in rows):raise ValueError('thermocouple column mismatch')
        warnings=['Measured thermocouples begin at nonuniform temperatures; do not silently replace with uniform 302.85 K. TC positions and junction identities require the author booklet.']
    elif source_id in ('ablantis_exp2_recession.dat','ablantis_exp2_temperature.dat'):
        angles=rows.pop(0);names=['time_s']+[f'angle_index_{i}' for i in range(len(angles))]
        if not rows or any(len(r)!=len(names) for r in rows):raise ValueError('angular matrix column mismatch')
        axes={'angle_values_as_published':angles,'angle_units':'UNRESOLVED' if 'temperature' in source_id else 'degree','values_unit':'K' if 'temperature' in source_id else 'mm_radius'}
        warnings=['Prescribed boundary data are not independent output validation when fed to the solver.','Published air recession time origin has a known transient-offset issue; retain raw time, do not silently shift.']
        if 'temperature' in source_id:warnings.append('Header says radians but values run 0 to 120.96376 and match degree-labelled recession coordinates. Unit conflict is unresolved; no automatic angle conversion.')
    else:raise ValueError('This source is a material/document artifact, not a declared numeric table importer')
    # NASA Cf first column ordering differs by source; retain exact source values.
    report={'status':'IMPORTED_NOT_VALIDATED','source':source,'columns':names,'axes':axes,'rows':rows,'original_headers':headers,'warnings':warnings,'claim':'Independent external reference import only. No native solver or experimental match has been demonstrated.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return {'status':report['status'],'rows':len(rows),'columns':len(names),'warnings':warnings,'output':str(output)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-id',required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--acknowledge-noncommercial-license',action='store_true');a=p.parse_args()
    print(json.dumps(import_source(a.source_id,a.input,a.output,a.acknowledge_noncommercial_license),indent=2))
if __name__=='__main__':main()
