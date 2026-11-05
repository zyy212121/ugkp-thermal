#!/usr/bin/env python3
"""Independent native/GPU implicit, gradient, boundary and periodic comparisons.

The operator timings printed by the executable are exploratory, without warmups
or repeated pairs; they do not establish whole-application performance.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', type=Path, required=True, help='CHMT case with system/solid/fvSchemes and fvSolution')
    p.add_argument('--binaries', type=Path, required=True, help='layer3_kernels.sh output')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--grids', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--mpi', action='store_true')
    args = p.parse_args(); args.work = args.work.resolve(); args.binaries = args.binaries.resolve()
    args.work.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'Independent original OpenFOAM operators and physical residual/heat-flux comparisons; isolated exploratory timings only.', 'cases': []}
    def save(): (args.work / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    def run(command, path):
        with path.open('w') as log:
            subprocess.run([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        for grid in args.grids:
            for shear in [0., .35]:
                case = args.work / f'grid{grid}-shear{shear:g}'; case.mkdir()
                shutil.copytree(args.template / 'system', case / 'system'); (case / 'constant').mkdir()
                vertices = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(shear,0,1),(1+shear,0,1),(1+shear,1,1),(shear,1,1)]
                mesh = ('FoamFile {version 2.0; format ascii; class dictionary; object blockMeshSolidDict;}\nconvertToMeters 1;\nvertices (' + ' '.join('('+ ' '.join(map(str,v)) +')' for v in vertices) + ');\n'
                    + f'blocks (hex (0 1 2 3 4 5 6 7) ({grid} {grid} {grid}) simpleGrading (1 1 1)); edges ();\n'
                    + """boundary (
left {type wall; faces ((0 4 7 3));}
right {type wall; faces ((1 2 6 5));}
bottom {type wall; faces ((0 3 2 1));}
top {type wall; faces ((4 5 6 7));}
front {type cyclic; neighbourPatch back; faces ((0 1 5 4));}
back {type cyclic; neighbourPatch front; faces ((3 7 6 2));}
); mergePatchPairs ();\n""")
                dictionary = case / 'system/blockMeshSolidDict'; dictionary.write_text(mesh)
                run(['blockMesh','-case',case,'-region','solid','-dict',dictionary],case/'mesh.log')
                record = {'grid':grid,'shear':shear,'status':'running'};report['cases'].append(record);save()
                for unit in ['solid_sparse','thermal_fields']:
                    run([args.binaries/unit,'-case',case],case/f'{unit}.log')
                    record[unit]=[line for line in (case/f'{unit}.log').read_text().splitlines() if line.startswith(('SPARSE_RESULT','FIELD_RESULT','PASS'))]
                record['status']='PASS';save()
        if args.mpi:
            case=args.work/'mpi';shutil.copytree(args.work/f'grid{args.grids[0]}-shear0.35',case)
            (case/'system/decomposeParDict').write_text('FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}\nnumberOfSubdomains 2; method simple; simpleCoeffs {n (2 1 1); delta 0.001;}\n')
            run(['decomposePar','-case',case,'-region','solid','-noFields'],case/'decompose.log')
            for unit in ['solid_sparse','thermal_fields']:
                run(['mpirun','--allow-run-as-root','-np','2',args.binaries/unit,'-case',case,'-parallel'],case/f'{unit}-mpi.log')
            report['mpi']='PASS: two partitions, processor exchange plus local cyclic interfaces'
        report['status']='PASS';save()
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));save();raise
    print(json.dumps({'status':report['status'],'report':str(args.work/'results.json')}))


if __name__=='__main__':main()
