#!/usr/bin/env python3
"""Nonuniform variable-property solid solves against the frozen native owner.

Timings include property/assembly/nonlinear work. With --repeats, discard one
warmup and measure repeated solves in the same resident owner. These are not
complete gas-solid application timings or alternating process pairs.
"""
import argparse
import json
import math
import statistics
from pathlib import Path
import shutil
import subprocess
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meshes',type=Path,required=True)
    p.add_argument('--binaries',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=0,help='warmup followed by repeated solves within each resident owner')
    p.add_argument('--grids',type=int,nargs='+',default=[8,32])
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','scope':__doc__,'timed_repeats':a.repeats,'cases':[]}
    def save():(a.work/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for grid in a.grids:
            for shear in ['0','0.35']:
                for aitken in [False,True]:
                    entry={'grid':grid,'shear':shear,'aitken':aitken,'runs':{}}
                    result['cases'].append(entry);save()
                    for kind in ['baseline','candidate']:
                        case=a.work/f'g{grid}-s{shear}-a{int(aitken)}-{kind}';case.mkdir()
                        seed=a.meshes/f'grid{grid}-shear{shear}'
                        for name in ['system','constant']:shutil.copytree(seed/name,case/name)
                        control=case/'system/controlDict'
                        control.write_text(control.read_text()+f'\nconductionAitken {str(aitken).lower()}; conductionInterval 0.05; conductionRepeats {a.repeats};\n')
                        (case/'system/solid/fvSolution').write_text('FoamFile {version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers {TsolidCandidate {solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 2000;}}\n')
                        start=time.monotonic()
                        with (case/'run.log').open('w') as log:
                            process=subprocess.run([str((a.binaries/kind).resolve()),'-case',str(case.resolve())],stdout=log,stderr=subprocess.STDOUT)
                        run={'exit_code':process.returncode,'process_seconds':time.monotonic()-start}
                        entry['runs'][kind]=run;save()
                        assert process.returncode==0,(kind,case,process.returncode)
                        samples=[x.split()[1:] for x in (case/'run.log').read_text().splitlines() if x.startswith('CONDUCTION_RESULT ')]
                        assert len(samples)==a.repeats+1
                        fields=samples[-1]
                        run['samples']=[dict(zip(['cells','iterations','solve_seconds','energy_residual','wall_flux_error','reported_energy_residual'],map(float,x))) for x in samples]
                        assert all(math.isfinite(v) for sample in run['samples'] for v in sample.values()),(kind,case,'non-finite diagnostic')
                        run['solve_samples_seconds']=[float(x[2]) for x in samples[1:]]
                        if a.repeats:run['median_solve_seconds']=statistics.median(run['solve_samples_seconds'])
                        run['warmup_seconds']=float(samples[0][2])
                        run.update(zip(['cells','iterations','solve_seconds','energy_residual','wall_flux_error','reported_energy_residual'],map(float,fields)))
                        values=[list(map(float,(case/f'conduction-values-{i}.txt').read_text().split())) for i in range(a.repeats+1)]
                        assert all(math.isfinite(x) for sample in values for x in sample),(kind,case,'non-finite temperature')
                        if kind=='baseline':reference=values
                        else:
                            assert all(len(x)==len(y) for x,y in zip(reference,values))
                            entry['maximum_temperature_error']=max(abs(x-y) for old,new in zip(reference,values) for x,y in zip(old,new))
                            assert entry['maximum_temperature_error']<1e-7,entry
                            if a.repeats:entry['median_speedup']=entry['runs']['baseline']['median_solve_seconds']/run['median_solve_seconds']
                        save()
        result['status']='PASS';save()
    except BaseException as error:
        result.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
