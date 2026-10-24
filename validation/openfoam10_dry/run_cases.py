from pathlib import Path
import subprocess, os, csv, json, shlex
ROOT=Path(__file__).resolve().parent
ENV=os.environ.get('OF10_ENV','')
ENV_PREFIX=f'. {shlex.quote(ENV)} >/dev/null 2>&1; ' if ENV else ''
def case(mode,n,dt,end,tag=''):
 d=ROOT/'cases'/f'{mode}_N{n}_dt{dt:g}{tag}';(d/'system').mkdir(parents=True,exist_ok=True);(d/'0').mkdir(exist_ok=True)
 threeD=mode.endswith('3d')
 temperature_dimension=1 if mode in ('coupled','surface') else 0
 side_boundary='type fixedValue; value uniform 0;' if threeD else 'type empty;'
 hdr=lambda cls,obj:f'FoamFile {{version 2.0; format ascii; class {cls}; object {obj};}}\n'
 (d/'system/blockMeshDict').write_text(hdr('dictionary','blockMeshDict')+f'''convertToMeters 1;
vertices ((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) ({n} {n if threeD else 1} {n if threeD else 1}) simpleGrading (1 1 1)); edges ();
boundary (left {{type patch; faces ((0 4 7 3));}} right {{type patch; faces ((1 2 6 5));}} sides {{type {"patch" if threeD else "empty"}; faces ((0 1 5 4)(3 7 6 2)(0 3 2 1)(4 5 6 7));}}); mergePatchPairs ();
''')
 (d/'system/controlDict').write_text(hdr('dictionary','controlDict')+f'application dryMaterialFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {end}; deltaT {dt}; writeControl timeStep; writeInterval 10000000; writeFormat ascii; writePrecision 15; runTimeModifiable false;\n')
 (d/'system/fvSchemes').write_text(hdr('dictionary','fvSchemes')+'''ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default Gauss linear;} laplacianSchemes {default Gauss linear orthogonal;} interpolationSchemes {default linear;} snGradSchemes {default orthogonal;} fluxRequired {default no; T;}\n''')
 (d/'system/fvSolution').write_text(hdr('dictionary','fvSolution')+'''solvers {T {solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; maxIter 10000;}}\n''')
 (d/'system/validationProperties').write_text(hdr('dictionary','validationProperties')+f'mode {mode}; length 1; dt {dt}; end {end};\n')
 left,right=(200000,100000) if mode=='darcy' else ((340,300) if mode=='coupled' else ((300,1000) if mode=='surface' else ((.1,.9) if mode=='mixture' else (0,0))))
 (d/'0/T').write_text(hdr('volScalarField','T')+f'dimensions [0 0 0 {temperature_dimension} 0 0 0]; internalField uniform 0; boundaryField {{left {{type fixedValue; value uniform {left};}} right {{type fixedValue; value uniform {right};}} sides {{{side_boundary}}}}}\n')
 with open(d/'run.log','w') as log:
  subprocess.run(['bash','-c',f'{ENV_PREFIX}blockMesh -case {shlex.quote(str(d))} && {shlex.quote(str(ROOT / "bin" / "dryMaterialFoam"))} -case {shlex.quote(str(d))}'],stdout=log,stderr=subprocess.STDOUT,check=True)
 return next(csv.DictReader(open(d/'result.csv')))
if __name__=='__main__':
 rows=[]
 for mode in ['conduction','diffusion']:
  for n in [20,40,80]: rows.append(case(mode,n,0.0005,0.5))
  for dt in [0.04,0.02,0.01]:rows.append(case(mode,400,dt,0.4))
 for n in [20,40,80]:rows.append(case('darcy',n,0.01,1))
 (ROOT/'results/pde_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
