from pathlib import Path
import shutil, subprocess, json, hashlib, sys

BASE=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr21-audit-execution-20261019')
SOURCE=Path('/home/lss/ARC3696-runs/pr19-f9e8d775-20261019T185625+0800/M0p55/baseline-lae7a4r6')
OUT=Path('/mnt/d/tempt/pr21-wallmodel-audit-20261019/of10-reference')
OUT.mkdir(parents=True,exist_ok=True)
def header(obj,cls='dictionary'):
 return 'FoamFile { version 2.0; format ascii; class '+cls+'; object '+obj+'; }\n'
walls=['upstreamWall','porousWall','downstreamWall']
def writecase(ny):
 case=BASE/f'of10-reference-{ny}'
 assert not case.exists(),case
 for sub in ['0','constant','system']:(case/sub).mkdir(parents=True)
 mesh=(SOURCE/'system/blockMeshDict').read_text().replace('80 1)',f'{ny} 1)')
 (case/'system/blockMeshDict').write_text(mesh)
 thermo=(SOURCE/'constant/fluidProperties').read_text().split('turbulence')[0]
 thermo=thermo.replace('object fluidProperties','object physicalProperties')
 (case/'constant/physicalProperties').write_text(thermo)
 (case/'constant/momentumTransport').write_text(header('momentumTransport')+'''simulationType RAS;
RAS { model kOmegaSST; turbulence on; printCoeffs on; }
''')
 (case/'constant/thermophysicalTransport').write_text(header('thermophysicalTransport')+'RAS { model eddyDiffusivity; Prt 0.9; }\n')
 values={'U':('volVectorField','[0 1 -1 0 0 0 0]','(179.81233357102013 0 0)','fixedValue; value uniform (0 0 0)'),
 'T':('volScalarField','[0 0 0 1 0 0 0]','275.90898782976797','fixedValue; value uniform 290.53216418474562'),
 'p':('volScalarField','[1 -1 -2 0 0 0 0]','124094.11053558919','zeroGradient'),
 'k':('volScalarField','[0 2 -2 0 0 0 0]','1.212467823909593','kqRWallFunction; value uniform 1.212467823909593'),
 'omega':('volScalarField','[0 0 -1 0 0 0 0]','2010.3630052883425','omegaWallFunction; blended false; beta1 0.075; value uniform 2010.3630052883425'),
 'nut':('volScalarField','[0 2 -1 0 0 0 0]','0','nutUSpaldingWallFunction; kappa 0.41; E 9.8; Cmu 0.09; tolerance 1e-10; maxIter 100; value uniform 0'),
 'alphat':('volScalarField','[1 -1 -1 0 0 0 0]','0','compressible::alphatJayatillekeWallFunction; Prt 0.9; kappa 0.41; E 9.8; Cmu 0.09; value uniform 0')}
 for name,(cls,dim,val,wallbc) in values.items():
  s=header(name,cls)+f'dimensions {dim};\ninternalField uniform {val};\nboundaryField {{\n'
  for patch in ['inlet','outlet','top']:
   bc='calculated; value uniform '+val if name in ['nut','alphat'] else ('zeroGradient' if patch=='outlet' and name!='p' else 'fixedValue; value uniform '+val)
   s+=f'{patch} {{ type {bc}; }}\n'
  for patch in walls:s+=f'{patch} {{ type {wallbc}; }}\n'
  s+='sides { type empty; }\n}\n'
  (case/'0'/name).write_text(s)
 (case/'system/controlDict').write_text(header('controlDict')+'''application rhoSimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 3000; deltaT 1;
writeControl timeStep; writeInterval 500; purgeWrite 0; writeFormat ascii;
writePrecision 16; writeCompression off; timeFormat general; timePrecision 8;
runTimeModifiable false;
functions {
 wallShear { type wallShearStress; libs ("libfieldFunctionObjects.so"); executeControl timeStep; executeInterval 50; writeControl writeTime; }
 yPlus { type yPlus; libs ("libfieldFunctionObjects.so"); executeControl timeStep; executeInterval 50; writeControl writeTime; }
 centers { type writeCellCentres; libs ("libfieldFunctionObjects.so"); writeControl writeTime; }
}
''')
 (case/'system/fvSchemes').write_text(header('fvSchemes')+'''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes {
 default none;
 div(phi,U) bounded Gauss linearUpwind grad(U);
 div(phi,e) bounded Gauss linearUpwind grad(e);
 div(phi,K) bounded Gauss linearUpwind grad(K);
 div(phi,Ekp) bounded Gauss linearUpwind grad(Ekp);
 div(phi,k) bounded Gauss upwind;
 div(phi,omega) bounded Gauss upwind;
 div(phid,p) Gauss upwind;
 div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''')
 (case/'system/fvSolution').write_text(header('fvSolution')+'''solvers {
 p { solver GAMG; smoother DIC; tolerance 1e-10; relTol 0.01; }
 "(U|e|k|omega)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.01; }
}
SIMPLE { nNonOrthogonalCorrectors 0;
 residualControl { p 1e-8; U 1e-8; e 1e-8; "(k|omega)" 1e-8; }
}
relaxationFactors {
 fields { p 0.3; rho 0.1; }
 equations { U 0.7; e 0.7; "(k|omega)" 0.7; }
}
''')
 # These are isolated diagnostic runs using installed OF10, not gasUGKP truth.
 (case/'reference-limitations.txt').write_text('Independent steady pureAir SST reference. Same domain/coarse mesh and physical inputs; OF10 pressure-based transport and OF10 wall functions. Not the same temporal trajectory. Standard OF10 energy equation omits explicit viscous-work transport, so use velocity/shear primarily and do not rank thermal fields from this alone.\n')
 return case

if __name__=='__main__':
 cases=[writecase(n) for n in map(int,sys.argv[1:] or ['80','160'])]
 manifest={str(case):{p.relative_to(case).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in case.rglob('*')if p.is_file()}for case in cases}
 (OUT/'input-hashes.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'cases':list(manifest),'original_inputs_unchanged':True},indent=2))
