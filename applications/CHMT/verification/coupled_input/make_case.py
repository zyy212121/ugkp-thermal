#!/usr/bin/env python3
"""Create a two-region thermal-contact input fixture; never a flow benchmark."""
from pathlib import Path
import hashlib
import sys
root=Path(sys.argv[1]);(root/'system/solid').mkdir(parents=True,exist_ok=True)
(root/'constant/solid').mkdir(parents=True,exist_ok=True)
(root/'0/solid').mkdir(parents=True,exist_ok=True)
def header(name,kind='dictionary'):
    return f'FoamFile {{ version 2.0; format ascii; class {kind}; object {name}; }}\n'
for name,x0,x1 in [('blockMeshDict',0,1),('blockMeshSolidDict',-1,0)]:
    (root/'system'/name).write_text(header(name)+f'''convertToMeters 1;
vertices (({x0} 0 0) ({x1} 0 0) ({x1} 1 0) ({x0} 1 0) ({x0} 0 1) ({x1} 0 1) ({x1} 1 1) ({x0} 1 1));
blocks (hex (0 1 2 3 4 5 6 7) (2 2 2) simpleGrading (1 1 1)); edges ();
boundary (
 left {{ type wall; faces ((0 4 7 3)); }}
 right {{ type wall; faces ((1 2 6 5)); }}
 sides {{ type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2)); }}
); mergePatchPairs ();
''')
(root/'system/controlDict').write_text(header('controlDict')+'''application CHMT;
startFrom startTime; startTime 0; stopAt endTime; endTime 0.01; deltaT 0.00001;
writeControl timeStep; writeInterval 100; writePrecision 17; runTimeModifiable false;
''')
schemes=header('fvSchemes')+'''fluxScheme Tadmor;
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; }
divSchemes { default none;
 "div(phi,U)" Gauss upwind; "div(phi,e)" Gauss upwind;
 "div(phi,K)" Gauss upwind; "div(phi,(p|rho))" Gauss upwind;
 "div(((rho*nuEff)*dev2(T(grad(U)))))" Gauss linear;
} laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; "reconstruct(.*)" upwind; }
snGradSchemes { default corrected; } fluxRequired { default no; }
'''
solution=header('fvSolution')+'''solvers { CHMTMaterialTemperature { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; } }
'''
for d in ('system','system/solid'):
    (root/d/'fvSchemes').write_text(schemes);(root/d/'fvSolution').write_text(solution)
gas='''gasMode mixtureFrozen; species (S0 S1);
speciesThermo {
 S0 { model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 3000; coefficients (1040 0 0); }
 S1 { model linearCp; molarMass 0.032; minTemperature 100; maxTemperature 3000; coefficients (1040 0 0); }
}
diffusion { model none; }
'''
(root/'constant/gasModelProperties').write_text(header('gasModelProperties')+gas)
properties='''schemaVersion 1; executionMode Multirate; modelName coupledInputFixture;
materialSource syntheticConstantSolid; mechanismSource noMaterialReaction;
condensedNames (C0 C1); elements ();
minDt 1e-12; maxDt 0.001; cfl 0.4;
gasConductivity 1; gasViscosity 0;
condensedThermo {
 C0 { rho 1000; cp0 1000; cp1 0; e0 0; conductivity 1; Tmin 100; Tmax 3000; element (); }
 C1 { rho 1000; cp0 1000; cp1 0; e0 0; conductivity 1; Tmin 100; Tmax 3000; element (); }
}
solidRegion solid;
surface { patch right; gasPatch left; }
multirate { couplingInterval 0.01; gasMaxDt 0.00001; }
'''
fingerprint=hashlib.sha256((properties+gas).encode()).hexdigest()
(root/'constant/chmtProperties').write_text(header('chmtProperties')+properties+f'modelFingerprint "{fingerprint}";\n')
def field(region,name,dimensions,value,vector=False):
    text=header(name,'volVectorField' if vector else 'volScalarField')
    text+=f'dimensions [{dimensions}];\ninternalField uniform {value};\nboundaryField {{\n'
    text+=''.join(f' {p} {{ type zeroGradient; }}\n' for p in ('left','right','sides'))+'}\n'
    (root/'0'/region/name).write_text(text)
for name,dims,value in [('rho','1 -3 0 0 0 0 0',1),('T','0 0 0 1 0 0 0',600),('Y_S0','0 0 0 0 0 0 0',1),('Y_S1','0 0 0 0 0 0 0',0)]:field('',name,dims,value)
field('','U','0 1 -1 0 0 0 0','(0 0 0)',True)
for name,dims,value in [('solidEnergyDensity','1 -1 -2 0 0 0 0',3e8),('porosity','0 0 0 0 0 0 0',0),('rho_C0','1 -3 0 0 0 0 0',1000),('rho_C1','1 -3 0 0 0 0 0',0),('rhoPore_S0','1 -3 0 0 0 0 0',0),('rhoPore_S1','1 -3 0 0 0 0 0',0)]:field('solid',name,dims,value)

if '--h2o2' in sys.argv[2:]:
    # The same actual two-region input, using the complete pinned public gas
    # chemistry fixture. No chemical time integration is claimed by preflight.
    import json
    import shutil
    data=Path(__file__).resolve().parents[4]/'common/chemistry/mechanisms'
    manifest=json.loads((data/'h2o2.manifest.json').read_text())
    names=manifest['species']
    for name in ('h2o2.gasModelProperties','h2o2.mechanism'):
        destination=root/'constant'/('gasModelProperties' if name.endswith('gasModelProperties') else name)
        if name.endswith('gasModelProperties'):destination.write_text(header('gasModelProperties')+(data/name).read_text())
        else:shutil.copyfile(data/name,destination)
    properties=properties.replace('elements ();','elements (O H Ar N);').replace('element ();','element (0 0 0 0);')
    gas=(root/'constant/gasModelProperties').read_text()
    fingerprint=hashlib.sha256((properties+gas).encode()).hexdigest()
    (root/'constant/chmtProperties').write_text(header('chmtProperties')+properties+f'modelFingerprint "{fingerprint}";\n')
    for name in names:
        field('','Y_'+name,'0 0 0 0 0 0 0',1 if name=='N2' else 0)
        field('solid','rhoPore_'+name,'1 -3 0 0 0 0 0',0)

if '--single' in sys.argv[2:]:
    if '--h2o2' in sys.argv[2:]:raise ValueError('single legacy fixture and reactive fixture are distinct')
    (root/'constant/materialGasProperties').write_text(header('materialGasProperties')+gas)
    (root/'constant/gasModelProperties').write_text(header('gasModelProperties')+'gasMode single;\n')
    properties+='singleGasSpecies S0;\n'
    fingerprint=hashlib.sha256((properties+gas+'gasMode single;').encode()).hexdigest()
    (root/'constant/chmtProperties').write_text(header('chmtProperties')+properties+f'modelFingerprint "{fingerprint}";\n')
