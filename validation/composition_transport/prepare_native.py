#!/usr/bin/env python3
"""Small generic native tests; these are not converged ARC3696 reproductions."""
from pathlib import Path
import sys,re,importlib.util,shutil,json
R=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('matrix',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(sys.argv[1]);out.mkdir(exist_ok=False,parents=True)
for name,flux in [('impermeable',[0.]*4),('uniform',[.1]*4),('nonuniform',[.01,.02,.1,.4])]:
 case=out/name;m.copy_fixture(m.HERE/'fixtures/gas-wall',case)
 for p in (case/'0').iterdir():
  text=p.read_text().replace('Y_A','Y_Air').replace('Y_B','Y_CO2');p.write_text(text)
  if p.name in ('Y_A','Y_B'):p.rename(p.with_name('Y_Air' if p.name=='Y_A' else 'Y_CO2'))
 gm=case/'constant/gasModelProperties';gm.write_text("""FoamFile { version 2.0; format ascii; class dictionary; object gasModelProperties; }
schemaVersion 1; gasMode mixtureFrozen; species (Air CO2);
speciesThermo {
 Air { model linearCp; molarMass .02896520682164515; minTemperature 1; maxTemperature 5000; coefficients (1004.675 0 0); }
 CO2 { model linearCp; molarMass .0440095; minTemperature 1; maxTemperature 5000; coefficients (819.27 0 0); }
}
diffusion { model constant; coefficients (0 0); turbulentSchmidt .7; }
""")
 shutil.copyfile(R/'validation/composition_transport/transport.dict',case/'constant/gasTransportProperties')
 fluid=case/'constant/fluidProperties';fluid.write_text(fluid.read_text().replace('nodes 8;','nodes 16; writeProfiles true;'))
 # Consistent rho from the configured EOS: initial and inlet Y=(.5,.5), T=300, p=101325.
 rho=101325/(8.31446261815324*(.5/.02896520682164515+.5/.0440095)*300)
 p=case/'0/rho';s=p.read_text();s=re.sub(r'((?:internalField|value) uniform )[^;]+',lambda a:a[1]+str(rho),s);p.write_text(s)
 p=case/'0/rhoE';energy=rho*((.5*1004.675+.5*819.27-101325/(rho*300))*300+.5*70**2);p.write_text(re.sub(r'((?:internalField|value) uniform )[^;]+',lambda a:a[1]+str(energy),p.read_text()))
 (case/'constant/gasWallProperties').write_text('FoamFile { version 2.0; format ascii; class dictionary; object gasWallProperties; }\npatches { wall { '+ ('speciesFlux (0 .1);' if name=='uniform' else 'speciesFluxFields { Air uniform 0; CO2 nonuniform List<scalar> 4 ('+' '.join(map(str,flux))+'); }')+' } }\n')
 (case/'system/controlDict').write_text("""FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application gasUGKPSteady; startFrom startTime; startTime 0; stopAt endTime; endTime 2;
deltaT 1; writeControl timeStep; writeInterval 1; purgeWrite 0;
writeFormat ascii; writePrecision 17; writeCompression off; timeFormat general; timePrecision 17;
runTimeModifiable false; adjustTimeStep false;
steadyGas { maxCo .01; maxDeltaT 1e-7; boundaryTime 0; relativeTolerance 0; absoluteTolerance 0;
minimumIterations 100; residualInterval 1; samplingDeltaT 1e-7; windowSamples 1;
minimumWindows 100; stableWindows 2; meanTolerance 0; inventoryTolerance 0; }
""")
 (case/'expected-injection.json').write_text(json.dumps(flux))
print(out)
