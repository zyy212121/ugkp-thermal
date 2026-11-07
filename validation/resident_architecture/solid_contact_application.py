#!/usr/bin/env python3
"""Actual three-region CHT execution and restart; no recording gas owner."""
import argparse,json,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/molecular_transport'))
import native_matrix as m
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
results=[]
for mode in ('constant','variable','limited'):
    seed=a.output/(mode+'-input');m.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact',seed,meta)
    for directory in ('0','system'):
        shutil.copytree(seed/directory/'solid',seed/directory/'aux')
    mesh=(seed/'system/blockMeshSolidDict').read_text()
    old=re.search(r'vertices\s*\((.*?)\);',mesh,re.S)
    vertices=[]
    for group in re.findall(r'\(([^()]*)\)',old[1]):
        xyz=list(map(float,group.split()));xyz[0]-=1;vertices.append('('+' '.join(format(x,'.17g') for x in xyz)+')')
    mesh=mesh[:old.start()]+'vertices ('+' '.join(vertices)+');'+mesh[old.end():]
    bm=seed/'system/blockMeshAuxDict';bm.write_text(mesh);m.run(['blockMesh','-case',seed,'-region','aux','-dict',bm],seed/'mesh-aux.log')
    t=seed/'0/solid/T';t.write_text(t.read_text().replace('uniform 300','uniform 500').replace('left{type zeroGradient;}','left{type fixedGradient;gradient uniform 0;value uniform 500;}'))
    props=('rho {type constant;value 1;} Cp {type constant;value 2;} kappa {type constant;value 10;} emissivity {type constant;value 0;}' if mode!='variable' else
        'rho {type table;outOfBounds error;values ((100 0.9)(350 1.1)(1000 1.5));} Cp {type table;outOfBounds error;values ((100 1)(300 1.5)(450 2)(1000 3));} kappa {type table;outOfBounds error;values ((100 5)(300 10)(500 20)(1000 40));} emissivity {type constant;value 0;}')
    (seed/'constant/solidRegionProperties').write_text('FoamFile {version 2.0;format ascii;class dictionary;object solidRegionProperties;}\nschemaVersion 1;\nsolidThermalCoupling {\n'+
        'formatVersion 2;fluidRegion region0;primarySolidRegion solid;solidRegions {solid {properties {'+props+'}} aux {properties {'+props+'}}}\n'+
        'fluidSolidInterfaces ({fluidPatch left;solidPatch right;solidRegion solid;mapping conservativeNonConformal;});\n'+
        'solidSolidInterfaces ({firstRegion aux;secondRegion solid;firstPatch right;secondPatch left;mapping oneToOneConformal;});\n'+
        f'solidInterfaceMaxIterations {1 if mode=="limited" else 100};solidInterfaceTemperatureRelativeTolerance 1e-10;solidInterfaceEnergyRelativeTolerance 1e-10;\n'+
        'couplingInterval .00002;solidContactCouplingInterval .00002;radiationEnabled false;\nnonlinearMaxIterations 100;nonlinearRelativeTolerance 1e-10;energyRelativeTolerance 1e-8;energyAbsoluteTolerance 1e-10;\n}\n')
    cases=[]
    for kind,solver in [('baseline',a.baseline),('candidate',a.candidate)]:
        case=a.output/(mode+'-'+kind);m.copy_input(seed,case);m.run([solver.resolve(),'-case',case],case/'run.log');cases.append(case)
    errors=m.compare_fields(m.times(cases[0])[-1],m.times(cases[1])[-1])
    restart=a.output/(mode+'-restart');m.copy_input(cases[1],restart);saved=m.times(cases[1])[0];shutil.copytree(saved,restart/saved.name);m.entry(restart/'system/controlDict','startTime',saved.name)
    m.run([a.candidate.resolve(),'-case',restart],restart/'run.log');restart_errors=m.compare_fields(m.times(cases[1])[-1],m.times(restart)[-1],restart=True)
    results.append(dict(case=mode,baseline_errors=errors,restart_errors=restart_errors));print(mode,'PASS',flush=True)
    (a.output/'results.json').write_text(json.dumps(dict(status='PASS',cases=results),indent=2)+'\n')
