#!/usr/bin/env python3
"""Paired full CHT radiation restart, preserving baseline output differences.

Internal physical fields, actual particle checkpoints and restarted CPU/GPU
outputs are checked independently. Existing continuous/restart differences in
calculated boundary fields are retained in the report, not silently discarded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import numpy as np
import layer4_coupled as coupled
layer=coupled.layer
matrix=coupled.matrix

def internal(path):
    text=path.read_text();match=re.search(r'internalField\s+uniform\s+([^;]+);',text)
    if match:return list(map(float,re.findall(layer.NUMBER,match[1]))),None
    match=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    assert match,path
    return list(map(float,re.findall(layer.NUMBER,match[2]))),int(match[1])

def compare_internal(old,new):
    errors={};boundaries=[]
    for path in old.rglob('*'):
        if not path.is_file() or path.suffix=='.dat':continue
        text=path.read_text()
        if 'internalField' not in text:continue
        other=new/path.relative_to(old);a,na=internal(path);b,nb=internal(other)
        if na is None:a=a*(nb or 1)
        if nb is None:b=b*(na or 1)
        assert len(a)==len(b),(path,na,nb)
        assert all(math.isfinite(x) for x in a+b),path
        error=max((abs(x-y)/max(1.,abs(x),abs(y)) for x,y in zip(a,b)),default=0)
        errors[str(path.relative_to(old))]=error;assert error<2e-12,(path,error)
        left=text.split('boundaryField',1)[1];right=other.read_text().split('boundaryField',1)[1]
        if left!=right:boundaries.append(str(path.relative_to(old)))
    return dict(maximum_scaled_internal_errors=errors,changed_boundary_output_fields=boundaries)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','reference','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--grid',type=int,default=2);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',scope=__doc__,cases=[])
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for threads in [1,4]:
            item=dict(angle_threads=threads,runs={});report['cases'].append(item);save();restarted={}
            for key,binary in [('baseline',a.baseline),('candidate',a.candidate)]:
                reference=a.reference/f'mobile1-threads{threads}-{key}';case=a.work/f'threads{threads}-{key}';matrix.copy_input(reference,case)
                times=matrix.times(reference);checkpoint=times[0];shutil.copytree(checkpoint,case/checkpoint.name)
                control=case/'system/controlDict';control.write_text(re.sub(r'startFrom\s+\w+;','startFrom latestTime;',control.read_text()))
                elapsed=matrix.run([binary.resolve(),'-case',case],case/'run.log');actual=matrix.times(case);assert len(times)==len(actual)
                record=dict(binary_sha256=layer.digest(binary),elapsed_seconds=elapsed,continuous_restart={})
                for old,new in zip(times,actual):record['continuous_restart'][old.name]=compare_internal(old,new)
                record['audit']=coupled.inventory(case,a.grid,True)
                left,right=coupled.contact.particles(times[-1]),coupled.contact.particles(actual[-1])
                differences={}
                for field in left:
                    x,y=left[field],right[field]
                    if np.array_equal(x,y,equal_nan=True):continue
                    assert x.dtype.kind=='f',field
                    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)),field
                    absolute=float(np.max(np.abs(x-y)));scaled=float(np.max(np.abs(x-y)/np.maximum(1.,np.maximum(np.abs(x),np.abs(y)))))
                    # The native implicit root accepts 1e-9 energy residuals.
                    # Restart reconstruction can redistribute roundoff into
                    # the stored enthalpy carry without changing temperature.
                    bound=1e-9 if field=='pThermalEnthalpyResidual' else 2e-12
                    assert scaled<bound,(field,scaled,bound)
                    differences[field]=dict(maximum_absolute_error=absolute,maximum_scaled_error=scaled,scaled_bound=bound)
                record['continuous_restart_particle_differences']=differences
                item['runs'][key]=record;restarted[key]=case;save()
            item['restarted_cpu_gpu_parity']=layer.compare_fields(restarted['baseline'],restarted['candidate'])
            for time,old in item['runs']['baseline']['continuous_restart'].items():
                new=item['runs']['candidate']['continuous_restart'][time]
                assert old['changed_boundary_output_fields']==new['changed_boundary_output_fields'],(time,old,new)
            left,right=(coupled.contact.particles(matrix.times(restarted[key])[-1]) for key in ['baseline','candidate'])
            assert all(np.array_equal(left[k],right[k],equal_nan=True) for k in left)
            item['restarted_cpu_gpu_particle_fields_bitwise_equal']=True
            item['baseline_restart_boundary_differences_preserved']=True;save()
        report['status']='PASS';save()
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
