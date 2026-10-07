#!/usr/bin/env python3
"""Small native OF10 meshes; all boundary values are set by the test adapter."""
import pathlib, sys
root=pathlib.Path(sys.argv[1]); nx=int(sys.argv[2]) if len(sys.argv)>2 else 12
(root/'constant'/'solid').mkdir(parents=True,exist_ok=True)
(root/'system'/'solid').mkdir(parents=True,exist_ok=True)
header=lambda obj:f'FoamFile {{ version 2.0; format ascii; class dictionary; object {obj}; }}\n'
for name,x0,x1,n in [('gas',0,.02,nx),('solid',-.004,0,max(2,nx//3))]:
    mesh=header('blockMeshDict')+f'''convertToMeters 1;
vertices (({x0} 0 0) ({x1} 0 0) ({x1} .01 0) ({x0} .01 0) ({x0} 0 .01) ({x1} 0 .01) ({x1} .01 .01) ({x0} .01 .01));
blocks (hex (0 1 2 3 4 5 6 7) ({n} 2 2) simpleGrading (1 1 1));
edges ();
boundary (
 left {{ type wall; faces ((0 4 7 3)); }}
 right {{ type wall; faces ((1 2 6 5)); }}
 sides {{ type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2)); }}
);
mergePatchPairs ();
'''
    (root/'system'/('blockMeshDict' if name=='gas' else 'blockMeshSolidDict')).write_text(mesh)
control=header('controlDict')+'''application chmtNativeOfCoupling;
startFrom startTime; startTime 0; stopAt endTime; endTime 0.0001; deltaT 0.0000002;
writeControl timeStep; writeInterval 100000; writePrecision 17; runTimeModifiable false;
'''
schemes=header('fvSchemes')+'''ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; "reconstruct(.*)" upwind; }
snGradSchemes { default corrected; }
fluxRequired { default no; }
'''
solution=header('fvSolution')+'''solvers {
 "(rho|rhoU|rhoE|rhoY.*)" { solver diagonal; }
 CHMTMaterialTemperature { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }
}
'''
(root/'system'/'controlDict').write_text(control)
for p in [root/'system',root/'system'/'solid']:
    (p/'fvSchemes').write_text(schemes); (p/'fvSolution').write_text(solution)
