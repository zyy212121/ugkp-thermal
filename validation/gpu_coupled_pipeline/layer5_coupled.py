#!/usr/bin/env python3
"""Uniformly refine a prepared CHMT input and compare fresh whole-process runs.

Preparation changes only mesh dimensions and an optional duration. Physics,
coupling interval, gas-step cap, integration and convergence controls are kept.
One warmup pair precedes alternating timed pairs; meshing and audits are outside
the timer. Numerical assertions reuse the established coupled/material oracles.
"""
import argparse,json,re,statistics
from pathlib import Path
import layer1,layer2_native
from repeat_case import input_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--inspector',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--scenario',default='receding')
    p.add_argument('--grid',type=int,default=2)
    p.add_argument('--end-time',type=float)
    p.add_argument('--repeats',type=int,default=0)
    a=p.parse_args();a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=False)
    layer1.matrix.ENV.update(CHMT_CUDA_STACK_BYTES='65536',OMP_NUM_THREADS='4')
    binaries={k:getattr(a,k).resolve() for k in ('baseline','candidate')}
    report={'status':'RUNNING','scenario':a.scenario,'grid':a.grid,'end_time':a.end_time,'binaries':{k:{'path':str(v),'sha256':layer1.digest(v)} for k,v in binaries.items()},'reference_input':str(a.reference.resolve()),'reference_input_sha256':input_hash(a.reference),'runs':[],'timing_scope':'Complete native process, including startup and normal output; mesh preparation and audit excluded; one warmup pair then alternating order.'}
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    reference=None;reference_material=None;cohort_hash=None;samples={k:[] for k in binaries};save()
    try:
        for repeat in range(a.repeats+1):
            for kind in (['baseline','candidate'] if repeat%2==0 else ['candidate','baseline']):
                case=a.work/f'{kind}-{repeat}';layer1.matrix.copy_input(a.reference,case)
                for name in ['blockMeshDict','blockMeshSolidDict']:
                    path=case/'system'/name;text,n=re.subn(r'(blocks\s*\(\s*hex\s*\([^)]*\)\s*)\(\d+\s+\d+\s+\d+\)',lambda m:m[1]+f'({a.grid} {a.grid} {a.grid})',path.read_text())
                    assert n==1,(path,'one uniform block expected');path.write_text(text)
                if a.end_time is not None:layer1.matrix.entry(case/'system/controlDict','endTime',a.end_time)
                layer1.matrix.run(['blockMesh','-case',case],case/'mesh.log')
                layer1.matrix.run(['blockMesh','-case',case,'-region','solid','-dict',case/'system/blockMeshSolidDict'],case/'mesh-solid.log')
                current_hash=input_hash(case)
                if cohort_hash is None:cohort_hash=current_hash
                assert current_hash==cohort_hash,'cohort input differs'
                row={'kind':kind,'repeat':repeat,'warmup':repeat==0,'case':str(case),'input_sha256':current_hash};report['runs'].append(row);save()
                row['elapsed_seconds']=layer1.matrix.run([binaries[kind],'-case',case],case/'run.log')
                material=layer2_native.checkpoint_material(case,a.inspector.resolve())
                if reference is None:reference=case;reference_material=material
                row['parity']=layer1.compare_chmt(reference,case)
                row['checkpoint_material_maximum_scaled_errors']=layer2_native.compare_material(reference_material,material)
                row['audit']=layer2_native.audit(case,a.scenario,material)
                row['status']='PASS'
                if repeat:samples[kind].append(row['elapsed_seconds'])
                save();print(json.dumps({'kind':kind,'repeat':repeat,'elapsed_seconds':row['elapsed_seconds'],'status':row['status']}),flush=True)
        medians={k:statistics.median(v) for k,v in samples.items() if v}
        report['performance']={'samples_seconds':samples,'median_seconds':medians}
        if len(medians)==2:report['performance']['speedup']=medians['baseline']/medians['candidate']
        report['status']='PASS';save()
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise

if __name__=='__main__':main()
