#!/usr/bin/env python3
"""Native discrete-ordinate owner comparison, including angular intensity exports.

The test-only instrumentation exports the actual production owner's intensity.
Timing includes this identical binary output, properties and all source/linear
iterations. It is not a complete coupled-application benchmark.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--table',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--grids',type=int,nargs='+',default=[8,16])
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args();a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':__doc__,'cases':[],'binaries':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('baseline',a.baseline),('candidate',a.candidate)] if v}}
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(command,log):
        with log.open('w') as output:r=subprocess.run(list(map(str,command)),stdout=output,stderr=subprocess.STDOUT)
        assert r.returncode==0,(r.returncode,log)
    try:
        for grid in a.grids:
            for shear in [0,.35]:
                for threads in [1,4]:
                    item={'grid':grid,'shear':shear,'angle_threads':threads,'runs':{}};report['cases'].append(item);save()
                    for kind,binary in [('baseline',a.baseline),('candidate',a.candidate)]:
                        if not binary:continue
                        case=a.work/f'g{grid}-s{shear}-t{threads}-{kind}';(case/'system').mkdir(parents=True);(case/'constant').mkdir()
                        def header(name):return f'FoamFile {{version 2.0; format ascii; class dictionary; object {name};}}\n'
                        (case/'system/controlDict').write_text(header('controlDict')+f'application radiationTest; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .1; writeControl timeStep; writeInterval 1; angleThreads {threads}; radiationRepeats {a.repeats};')
                        (case/'system/fvSchemes').write_text(header('fvSchemes')+'ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}')
                        (case/'system/fvSolution').write_text(header('fvSolution')+'solvers {}')
                        points=[(0,0,0),(.02,0,0),(.02,.01,0),(0,.01,0),(0,0,.01),(.02,0,.01),(.02,.01,.01),(0,.01,.01)]
                        vertices=' '.join(f'({x+shear*y} {y} {z})' for x,y,z in points)
                        (case/'system/blockMeshDict').write_text(header('blockMeshDict')+f'convertToMeters 1; vertices ({vertices}); blocks (hex (0 1 2 3 4 5 6 7) ({grid} {grid} {grid}) simpleGrading (1 1 1)); edges (); boundary (left {{type wall; faces ((0 4 7 3));}} right {{type wall; faces ((1 2 6 5));}} sides {{type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2));}}); mergePatchPairs ();')
                        side='specular' if shear==0 else 'diffuseGrey'
                        (case/'constant/radiationProperties').write_text(header('radiationProperties')+'sourceRelaxation 1; linearSolverTolerance 1e-12; linearSolverRelTol 0; linearSolverMaxIterations 1000; IBoundaryCondition diffuseGrey; boundaryAbsorptivity .8; boundaryRadiation {left {type blackBody; T 1000;} sides {type '+side+'; T 450;}}')
                        shutil.copyfile(a.table,case/'constant/mie.table');run(['blockMesh','-case',case],case/'mesh.log')
                        run([binary.resolve(),'-case',case],case/'run.log')
                        samples=[list(map(float,x.split()[1:])) for x in (case/'run.log').read_text().splitlines() if x.startswith('RADIATION_RESULT ')]
                        assert len(samples)==a.repeats+1 and all(math.isfinite(v) for row in samples for v in row)
                        item['runs'][kind]={'samples':samples};save()
                        fields=[];angles=[]
                        for i in range(a.repeats+1):
                            fields.append(list(map(float,(case/f'radiation-values-{i}.txt').read_text().split())))
                            values=array.array('d');values.frombytes((case/f'radiation-directions-{i}.bin').read_bytes());angles.append(values)
                        assert all(math.isfinite(v) for seq in fields+angles for v in seq)
                        if kind=='baseline':reference=(fields,angles)
                        else:
                            for name,old,new in zip(['fields','angles'],reference,[fields,angles]):
                                assert all(len(x)==len(y) for x,y in zip(old,new))
                                error=max(abs(x-y)/max(1.,abs(x),abs(y)) for u,v in zip(old,new) for x,y in zip(u,v))
                                item['maximum_scaled_'+name+'_error']=error;assert error<2e-10,(case,name,error)
                        save()
        report['status']='PASS';save()
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
