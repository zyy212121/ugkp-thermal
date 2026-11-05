#!/usr/bin/env python3
"""Compare the actual CHT temporary-solid owner against frozen pre-migration code."""
import argparse,json,hashlib,shutil,subprocess,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--meshes',type=Path,required=True);p.add_argument('--binaries',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':__doc__,'cases':[],'binaries':{k:hashlib.sha256((a.binaries/f'owner-{k}').read_bytes()).hexdigest() for k in ['baseline','candidate']}}
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for seed in sorted(p for p in a.meshes.iterdir() if p.is_dir()):
        case=a.work/seed.name;case.mkdir()
        for folder in ['system','constant','0']:shutil.copytree(seed/folder,case/folder)
        (case/'system/solid/fvSolution').write_text('FoamFile {version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers {TsolidCandidate {solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 2000;}}\n')
        row={'case':seed.name,'runs':{}};report['cases'].append(row);save()
        for label in ['baseline','candidate']:
            log=case/(label+'.log')
            with log.open('w') as f:r=subprocess.run([str(a.binaries/f'owner-{label}'),'-case',str(case)],stdout=f,stderr=subprocess.STDOUT)
            lines=log.read_text().splitlines()
            cells=[float(l.split()[-1]) for l in lines if l.startswith('OWNER_CELL ')]
            flux=[float(l.split()[-1]) for l in lines if l.startswith('OWNER_FLUX ')]
            stats=[list(map(float,l.split()[1:])) for l in lines if l.startswith('OWNER_RESULT ')]
            row['runs'][label]={'exit_code':r.returncode,'temperature':cells,'boundary_flux':flux,'iterations_energy_residual':stats};save()
            assert r.returncode==0 and cells and flux and len(stats)==1,(case,label)
            assert all(math.isfinite(x) for x in cells+flux+stats[0])
        left,right=row['runs']['baseline'],row['runs']['candidate'];assert len(left['temperature'])==len(right['temperature']) and len(left['boundary_flux'])==len(right['boundary_flux'])
        row['temperature_error']=max(abs(x-y) for x,y in zip(left['temperature'],right['temperature']))
        row['boundary_flux_error']=max(abs(x-y) for x,y in zip(left['boundary_flux'],right['boundary_flux']))
        assert row['temperature_error']<2e-8 and row['boundary_flux_error']<2e-8,row
        if seed.name in ('2-fixedJump','5-uniformJump'):
            assert abs(right['temperature'][0]-298)<1e-10 and abs(right['temperature'][1]-302)<1e-10
        row['status']='PASS';save();print(seed.name,'PASS',flush=True)
    report['status']='PASS';save()
if __name__=='__main__':main()
