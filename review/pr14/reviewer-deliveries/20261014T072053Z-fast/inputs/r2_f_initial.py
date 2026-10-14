import re,shutil,json
from r2_runner import *
from r2_native import save_case
# Prerequisite directed original-source tests are complete before the remaining gaps.
tests={x['id']:x for x in json.loads((D/'results.json').read_text())['tests']}
assert tests['CHEM-OWNER-S203']['verdict']=='PASS' and tests['NATIVE-CHT-coldWall2D-INPUT2']['verdict']=='PASS' and tests['CHMT-chemistry-full']['verdict']=='PASS'
for app in ['CHT','FSH']:
 b=ROOT/'work/fixtures'/('thin-real-owner-'+app);lib=ROOT/'work/cuda-actual-fault-observer.so'
 t=run('SST22-ACTUAL-OWNER-'+app,cmd('env','LD_PRELOAD='+str(lib),b,'sst'),[4,12,17,18,26,28],expected='Actual22 owner lateconfigure/reconfigure/nativeGPUinitialization, each22real allocationOOM/resume,actualcopyerror/resume; onlymissingowner reallocated; published canonicalall22; release.',inputs=[D/'inputs'/('thin-real-owner-'+app+'.cu'),D/'inputs/cuda-actual-fault-observer.cpp'],binary=b,variant='diagnostic-real-resource',kind='DIAGNOSTIC')
 raw=(D/t['log']).read_text();assertion(t,'PASS actualGPU SST22' in raw and raw.count('PASS SST_ACTUAL_ALLOC_RETRY')==22,raw)
src=OLD/'inputs/native/chmt/thermal-full';mesh=OLDROOT/'work/cases/chmt/thermal-full'
for name in ['SST-RK','moving-GCL','moving-chemistry']:
 original=OLD/'inputs/native/chmt/chemistry-full' if name=='moving-chemistry' else src
 case=ROOT/'work/cases/extra'/name;shutil.copytree(original,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(mesh/rel,case/rel)
 prop=case/'constant/chmtProperties';text=prop.read_text()
 if name=='SST-RK':
  text+='\nenableSst true; sst { wallTreatment lowRe; }\n';text=re.sub(r'gasViscosity\s+[^;]+;','gasViscosity 1e-5;',text)
  for field,value,dimensions in [('k','.1','[0 2 -2 0 0 0 0]'),('omega','10','[0 0 -1 0 0 0 0]')]:
   q=case/'0'/field;template=(case/'0/T').read_text();template=re.sub(r'object\s+T;', 'object '+field+';',template);template=re.sub(r'dimensions\s+\[[^]]+\];','dimensions '+dimensions+';',template);template=re.sub(r'internalField\s+uniform\s+[^;]+;','internalField uniform '+value+';',template);template=re.sub(r'\bvalue\s+uniform\s+[^;]+;','value uniform '+value+';',template);q.write_text(template)
 else:text+='\nmeshMotion { policy PrescribedSinusoidal; amplitude (.001 0 0); spatialWaveNumber (1 0 0); spatialOrigin (0 0 0); angularFrequency 100; timeOrigin 0; }\n'
 prop.write_text(text)
 inp=D/'inputs/native/extra'/name;shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 binary=ROOT/('work/install/ns10/bin/CHMT' if name=='moving-chemistry' else 'work/install/ns2/bin/CHMT')
 t=run('EXTRA-'+name,cmd(binary,'-case',case),[8,9,10,14,22,26,29],expected='Actualnative original300s shortcase: activeSST/RK or nonuniform movingmesh withoriginal chemistry; fullrealstage/geometric/budget/checkpoint outputs retained, separate audit required.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=binary)
 if (case/'chmtOutput').is_dir():save_case(case,name,'extra')
 amend(t['id'],verdict='INCONCLUSIVE' if t['exit_code']==0 else 'FAIL',observations='Actual native scope executed; independent full budget/stagevolume/GCL audit pending.',artifacts=['artifacts/extra/'+name])
