#!/usr/bin/env python3
import argparse,json,shutil,subprocess
from pathlib import Path

def parse(path):
    cells={};windows={}
    for line in path.read_text().splitlines():
        words=line.split()
        if words[:1]==["CONTACT_CELL"]:cells[tuple(words[1:4])]=float(words[4])
        elif words[:1]==["CONTACT_WINDOW"]:windows[words[1]]=(int(words[2]),float(words[3]))
    if len(windows)!=4 or not cells:raise RuntimeError(f"incomplete output: {path}")
    return cells,windows

def main():
    p=argparse.ArgumentParser();p.add_argument('--meshes',type=Path,required=True);p.add_argument('--binaries',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False);results=[]
    for seed in sorted(a.meshes.iterdir()):
        if not seed.is_dir():continue
        for mode,options in [('constant',[]),('variable',['-variable-properties']),('variable-aitken',['-variable-properties','-aitken'])]:
            case=a.work/(seed.name+'-'+mode);shutil.copytree(seed,case)
            for folder in ('system','constant','0'):shutil.copytree(case/folder/'solid',case/folder/'auxiliary')
            # Preserve the original native discretization and tolerances.
            solution='FoamFile {version 2.0; format ascii; class dictionary; object fvSolution;}\nsolvers {".*" {solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; maxIter 1000;}}\n'
            for region in ('solid','auxiliary'):(case/'system'/region/'fvSolution').write_text(solution)
            parsed=[]
            for kind in ('baseline','candidate'):
                log=case/(kind+'.log')
                with log.open('w') as out:subprocess.run([str(a.binaries/('contact-'+kind)),'-case',str(case),*options],stdout=out,stderr=subprocess.STDOUT,check=True)
                parsed.append(parse(log))
            (bc,bw),(cc,cw)=parsed
            if bc.keys()!=cc.keys():raise RuntimeError('cell extent mismatch')
            error=max(abs(bc[k]-cc[k]) for k in bc);energy=max(abs(v[1]) for v in cw.values())
            if error>2e-8 or energy>2e-8:raise RuntimeError(f'contact comparison failed: {case} {error} {energy}')
            results.append(dict(case=case.name,maxTemperatureDifference=error,maxEnergyResidual=energy,baselineWindows=bw,candidateWindows=cw,pairedCells=len(bc)))
    (a.work/'results.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
    print('PASS contact native comparisons',len(results),'max temperature',max(x['maxTemperatureDifference'] for x in results))
if __name__=='__main__':main()
