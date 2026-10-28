#!/usr/bin/env python3
"""Native OpenFOAM/CUDA jump regression; reports local equations, not only sums."""
from pathlib import Path
import argparse,subprocess,json,hashlib,math

def make_case(case,kind,jump=0,grid=(1,2,1),shear=0):
    (case/'constant').mkdir(parents=True);(case/'0/solid').mkdir(parents=True)
    (case/'system/solid').mkdir(parents=True)
    header=lambda obj,cls='dictionary':f'FoamFile {{version 2.0; format ascii; class {cls}; object {obj};}}\n'
    (case/'system/controlDict').write_text(header('controlDict')+'application jumpProbe; startFrom startTime; startTime 0; stopAt endTime; endTime 4; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17; runTimeModifiable false;\n')
    schemes=header('fvSchemes')+'ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;}\n'
    for prefix in [case/'system',case/'system/solid']:
        (prefix/'fvSchemes').write_text(schemes);(prefix/'fvSolution').write_text(header('fvSolution')+'solvers {}\n')
    vertices=[(0,0,0),(1,0,0),(1,2,0),(0,2,0),(shear,0,1),(1+shear,0,1),(1+shear,2,1),(shear,2,1)]
    mesh=header('blockMeshDict')+'convertToMeters 1; vertices ('+' '.join('('+' '.join(map(str,v))+')' for v in vertices)+');\n'
    mesh+=f'blocks (hex (0 1 2 3 4 5 6 7) ({grid[0]} {grid[1]} {grid[2]}) simpleGrading (1 1 1)); edges ();\n'
    mesh+="""boundary (
left {type wall; faces ((0 4 7 3));}
right {type wall; faces ((1 2 6 5));}
bottom {type wall; faces ((0 3 2 1));}
top {type wall; faces ((4 5 6 7));}
front {type cyclic; neighbourPatch back; faces ((0 1 5 4));}
back {type cyclic; neighbourPatch front; faces ((3 7 6 2));}
); mergePatchPairs ();\n"""
    (case/'system/blockMeshSolidDict').write_text(mesh)
    patch='type cyclic;'
    if kind=='fixedJump':patch=f'type fixedJump; patchType cyclic; jump uniform {jump}; value uniform 300;'
    if kind=='uniformJump':patch=f'type uniformJump; patchType cyclic; jumpTable constant {jump}; value uniform 300;'
    if kind=='varying':patch='type uniformJump; patchType cyclic; jumpTable table ((0 10)(1 -5)(2 20)(3 0)); value uniform 300;'
    if kind=='nonuniform':
        n=grid[0]*grid[2];values=[10*(-1 if i%2 else 1)+i for i in range(n)]
        patch=f'type fixedJump; patchType cyclic; jump nonuniform List<scalar> {n} ('+' '.join(map(str,values))+'); value uniform 300;'
    (case/'0/solid/T').write_text(header('T','volScalarField')+'dimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField {left {type fixedGradient; gradient uniform 0; value uniform 300;} right {type fixedGradient; gradient uniform 0; value uniform 300;} bottom {type zeroGradient;} top {type zeroGradient;} front {'+patch+'} back {'+patch+'}}\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binaries',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    a.work.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','scope':__doc__,'cases':[],'binaries':{k:hashlib.sha256((a.binaries/f'jump-{k}').read_bytes()).hexdigest() for k in ['before','after']}}
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    cases=[('cyclic',0,(1,2,1),0),('fixedJump',0,(1,2,1),0),('fixedJump',10,(1,2,1),0),('fixedJump',-10,(1,2,1),0),('uniformJump',0,(1,2,1),0),('uniformJump',10,(1,2,1),0),('varying',10,(1,2,1),0),('nonuniform',10,(3,2,2),0),('nonuniform',10,(3,2,2),.3)]
    for index,(kind,jump,grid,shear) in enumerate(cases):
        case=a.work/f'{index}-{kind}';make_case(case,kind,jump,grid,shear)
        with (case/'mesh.log').open('w') as log:subprocess.run(['blockMesh','-case',str(case),'-region','solid','-dict',str(case/'system/blockMeshSolidDict')],stdout=log,stderr=subprocess.STDOUT,check=True)
        row={'kind':kind,'jump':jump,'grid':grid,'shear':shear,'runs':{}};report['cases'].append(row);save()
        for version in ['before','after']:
            log=case/(version+'.log')
            with log.open('w') as stream:result=subprocess.run([str(a.binaries/f'jump-{version}'),'-case',str(case)],stdout=stream,stderr=subprocess.STDOUT)
            lines=log.read_text().splitlines();samples=[list(map(float,l.split()[1:])) for l in lines if l.startswith('JUMP_RESULT ')]
            cells=[list(map(float,l.split()[1:])) for l in lines if l.startswith('CELL ')]
            row['runs'][version]={'exit_code':result.returncode,'samples':samples,'first_cell_values':[r for r in cells if r[0]==0 and r[1]==0]};save()
            assert all(math.isfinite(x) for r in samples+cells for x in r),(case,version,'non-finite output')
            assert len(samples)==8,(case,version,'incomplete probe',result.returncode)
            if version=='after':assert result.returncode==0 and all(s[-1]==1 for s in samples),(case,version)
            elif kind in ['cyclic'] or jump==0:assert result.returncode==0,(case,'zero-jump regression')
            else:assert result.returncode==1 and any(s[-1]==0 for s in samples),(case,'old defect not reproduced')
        if kind in ['fixedJump','uniformJump'] and jump==10 and grid==(1,2,1):
            rows=row['runs']['after']['first_cell_values'];assert abs(rows[0][3]-298)<1e-10 and abs(rows[1][3]-302)<1e-10 and abs(rows[0][4]-298)<1e-10 and abs(rows[1][4]-302)<1e-10,rows
        row['status']='PASS';save();print(case.name,'PASS',flush=True)
    report['status']='PASS';save()
if __name__=='__main__':main()
