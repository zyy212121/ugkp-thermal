import importlib.util,json,sys,re,struct
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('m',R/'validation/transient_particles/check_native.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report={}
for solver in ['gasUGKP','gasUGKPSteady']:
 source=Path(__file__).resolve().parent/'fixtures'/('gas-table' if solver=='gasUGKP' else 'steady-table');case=out/solver;m.copy_inputs(source,case)
 p=case/'0/epsilonS';p.write_text(p.read_text().replace('internalField uniform 0;','internalField nonuniform List<scalar> 16 ('+' '.join(['5.357142857142857e-10']+['0']*15)+');'))
 m.set_entry(case/'constant/particleProperties','parcelMass','1e-10')
 m.set_entry(case/'constant/schedulingProperties','gpuResidentPureGasOnly','false')
 m.set_entry(case/'constant/schedulingProperties','gpuResidentParticleCapacity','50000')
 for key,value in [('particleTemperatureTransport','true'),('particleRho','2800'),('particleCp','800'),('particleTMin','1'),('particleTMax','10000')]:m.set_entry(case/'constant/particleProperties',key,value)
 p=case/'constant/fluidProperties';s=p.read_text().replace('wallTreatment boundaryLayer','wallTreatment lowRe');s=re.sub(r'boundaryLayer\s*\{[^}]*\}','',s);p.write_text(s)
 (case/'0/Tp').write_text('FoamFile {version 2.0;format ascii;class volScalarField;object Tp;}\ndimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField { inlet {type fixedValue;value uniform 300;} outlet {type zeroGradient;} wall {type zeroGradient;} top {type symmetryPlane;} sides {type empty;} }\n')
 with (case/'0/gpuResidentStrictParticles.dat').open('wb') as f:
  f.write(b'UGKP_PARTICLES_SCHEMA1_BIN 1 1\n');f.write(struct.pack('<I',1))
  for v in [.25,.005,.005,70,0,0,500,0,2e-4,1e-10]:f.write(struct.pack('<d',v))
  f.write(struct.pack('<iiQQ',0,1,12345,1))
 row=m.run(R/'bin'/solver,case);assert row['exit']==0,(case,row)
 last=m.dirs(case)[-1];particles=m.parcel(last/'gpuResidentStrictParticles.dat');assert particles
 row['parcels']=len(particles);row['temperature_range']=[min(x[6] for x in particles),max(x[6] for x in particles)]
 assert all(all(__import__('math').isfinite(v) for v in x[:10]) for x in particles)
 report[solver]=row;print(solver,len(particles),'parcels PASS',flush=True)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
