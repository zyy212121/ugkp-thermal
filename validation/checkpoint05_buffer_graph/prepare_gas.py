"""Prepare-only fixed16 live-particle gas Graph inputs. No solver executes here."""
from pathlib import Path
import argparse,ast,types,json,hashlib,shutil,sys,re,os,struct

def prepare(source,output,baseline32,baseline64,candidate32,candidate64,mesh_donor):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source/'validation/solver_controls'));import migrate_case as m
    # Reuse only reviewed legacy helper definitions. Never import its executable CLI.
    original=source/'validation/resident_architecture/gas_graph.py';tree=ast.parse(original.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('entry','patch','particles')]
    namespace=dict(re=re,struct=struct,Path=Path);exec(compile(ast.Module(body=selected,type_ignores=[]),str(original),'exec'),namespace)
    entry,patch,particles=(namespace[k] for k in ('entry','patch','particles'))
    inputs=[]
    for solver in ('gasUGKP','gasUGKPSteady'):
      for csr in ('L1','auto'):
       for bits in (32,64):
        for variant in ('baseline','candidate'):
         label=f'{solver}-{csr}-{bits}-{variant}';case=output/label
         shutil.copytree(source/'validation/shared_boundary/fixtures/gas-total',case)
         donor=Path(mesh_donor)
         # Exact existing mesh definition; do not regenerate a different case.
         normalize=lambda s:re.sub(r'\s+','',re.sub(r'//[^\n]*|/\*[\s\S]*?\*/','',s))
         assert normalize((case/'system/blockMeshDict').read_text())==normalize((donor/'system/blockMeshDict').read_text())
         if not (case/'constant/polyMesh/points').exists():shutil.copytree(donor/'constant/polyMesh',case/'constant/polyMesh',dirs_exist_ok=True)
         (case/'constant/gasWallProperties').unlink(missing_ok=True)
         fp=case/'constant/fluidProperties';fp.write_text(re.sub(r'turbulence\s*\{[\s\S]*','turbulence { simulationType laminar; }\n',fp.read_text()))
         namespace['a']=types.SimpleNamespace(csr_auto=csr=='auto',pressure_projection=False,csr_inspection_interval=1)
         particles(case)
         steady=solver=='gasUGKPSteady'
         for key,value in [('deltaT',1 if steady else '1e-8'),('endTime',6 if steady else '4e-7'),('writeControl','timeStep'),('writeInterval',1)]:entry(case/'system/controlDict',key,value)
         patch(case,'T','inlet','type uniformFixedValue; uniformValue table ((0 300) (1e-6 400)); value uniform 300;')
         m.migrate_case(case)
         (case/'constant/precisionProperties').write_text(m.HEADER.replace('object scheme','object precisionProperties')+f'precision {bits};\n')
         for key,value in [('adjustTimeStep','false' if steady else 'true'),('maxCo','.1'),('maxDeltaT','1e-7'),('maximumTimeStepGrowth','1.3'),('courantUpdateInterval','1')]:m.set_control(case,'gasTimeStep',key,value)
         m.set_control(case,'gasNumerics.ddtSchemes','default','SSPRK2')
         if steady:
          for key,value in [('boundaryTime','1.25e-7'),('minimumIterations','100'),('relativeTolerance','0'),('absoluteTolerance','0'),('residualInterval','1')]:m.set_control(case,'steadyGas',key,value)
         binaries=locals()[variant+str(bits)]
         inputs.append(dict(id=label,case=str(case),solver=solver,precision=bits,variant=variant,csr=csr,argv=[str(Path(binaries)/solver),'-case',str(case)],liveParticleCount=512 if csr=='auto' else 1,physicalSource='existing gas_graph.py live-particle laminar frozen-mixture SSPRK2 path',scope='Gas/particle graph physical regression, not turbulent wall acceptance'))
    manifest=dict(status='FROZEN_NOT_RUN',processes=inputs,process_count=16,source=str(source),input_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()},scope='One fixed16 baseline/candidate32/64 transient+steady L1/auto; no solver launched by preparation')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--mesh-donor',required=True)
 for n in ('baseline32','baseline64','candidate32','candidate64'):p.add_argument('--'+n,required=True)
 a=p.parse_args();prepare(**vars(a))
