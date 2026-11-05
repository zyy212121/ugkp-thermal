#!/usr/bin/env python3
"""Alternating native process timings on one already validated radiation input.

One unmeasured warmup per binary precedes the measured pairs. Mesh construction
is excluded, process startup and ordinary output included. The radiation-owner
probe additionally includes its identical test-only angular export.
"""
import argparse
import array
import json
import math
from pathlib import Path
import statistics
import shutil
import layer4_coupled as coupled
layer=coupled.layer
matrix=coupled.matrix

def angular(path):
    data=array.array('d');data.frombytes(path.read_bytes());return data

def radiation_parity(reference,case):
    maximum=0.
    for pattern,reader in [('radiation-values-*.txt',lambda p:list(map(float,p.read_text().split()))),('radiation-directions-*.bin',angular)]:
        paths=sorted(reference.glob(pattern));assert paths
        for path in paths:
            old,new=reader(path),reader(case/path.name);assert len(old)==len(new)
            for a,b in zip(old,new):
                assert math.isfinite(a) and math.isfinite(b)
                maximum=max(maximum,abs(a-b)/max(1.,abs(a),abs(b)))
    assert maximum<2e-10,(case,maximum)
    return maximum

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','reference','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--solver',choices=['radiation','CHT'],default='radiation')
    p.add_argument('--grid',type=int,default=2);p.add_argument('--mobile',action='store_true');a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',scope=__doc__,reference=str(a.reference),solver=a.solver,runs=[],binaries={key:layer.digest(binary) for key,binary in [('baseline',a.baseline),('candidate',a.candidate)]})
    if a.solver=='CHT':report['reference_conservation']=coupled.inventory(a.reference,a.grid,a.mobile,enforce_absolute_bounds=False)
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for repeat in range(a.repeats+1):
            order=[('baseline',a.baseline),('candidate',a.candidate)]
            if repeat%2:order.reverse()
            for key,binary in order:
                case=a.work/f'{repeat}-{key}';case.mkdir()
                for name in ['0','constant','system']:
                    if (a.reference/name).exists():shutil.copytree(a.reference/name,case/name)
                record=dict(repeat=repeat,kind=key,warmup=repeat==0,elapsed_seconds=matrix.run([binary.resolve(),'-case',case],case/'run.log'))
                if a.solver=='CHT':
                    record['parity']=layer.compare_fields(a.reference,case);record['audit']=coupled.inventory(case,a.grid,a.mobile,enforce_absolute_bounds=False)
                    for metric in ['maximum_transaction_model_energy_residual_J','maximum_nonlinear_enthalpy_inventory_drift_J','maximum_closed_mass_residual_kg']:
                        assert abs(record['audit'][metric]-report['reference_conservation'][metric])<2e-10,(case,metric)
                else:
                    record['maximum_scaled_field_and_angle_error']=radiation_parity(a.reference,case)
                    samples=[list(map(float,row.split()[1:])) for row in (case/'run.log').read_text().splitlines() if row.startswith('RADIATION_RESULT ')]
                    assert len(samples)==3;record['owner_seconds']=[row[1] for row in samples]
                    record['source_iterations']=[int(row[2]) for row in samples]
                    record['conservation_residual_W']=[row[6] for row in samples]
                report['runs'].append(record);save()
        summary={}
        for key in ['baseline','candidate']:
            values=[row['elapsed_seconds'] for row in report['runs'] if row['kind']==key and not row['warmup']]
            summary[key]=dict(samples=values,median=statistics.median(values),minimum=min(values),maximum=max(values))
        summary['process_speedup']=summary['baseline']['median']/summary['candidate']['median']
        if a.solver=='radiation':
            summary['owner_medians']={key:[statistics.median(row['owner_seconds'][i] for row in report['runs'] if row['kind']==key and not row['warmup']) for i in range(3)] for key in ['baseline','candidate']}
        status='PASS'
        if a.solver=='CHT' and not all(report['reference_conservation'][k] for k in ['absolute_energy_bound_pass','absolute_mass_bound_pass']):
            status='PASS_PARITY_WITH_BASELINE_CONSERVATION_LIMIT'
        report.update(status=status,timing=summary);save()
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
