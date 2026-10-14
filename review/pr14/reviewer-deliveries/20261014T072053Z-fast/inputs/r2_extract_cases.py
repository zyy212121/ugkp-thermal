import shutil,json
from r2_runner import *
from r2_native import compare_dirs
P=pathlib.Path(__file__).parent
for name in ['SST-RK','moving-GCL','moving-chemistry']:
 case=ROOT/'work/cases/extra'/name;dest=D/'artifacts/extra'/name;shutil.copytree(case/'chmtOutput',dest,dirs_exist_ok=True)
src=D/'inputs/checkpoint_actual_reader.cpp';shutil.copy2(P/'r2_checkpoint_dump.cpp',src)
for ns,names in [(2,['SST-RK','moving-GCL']),(10,['moving-chemistry'])]:
 binary=ROOT/'work/fixtures'/('checkpoint-ASan-'+str(ns))
 t=run('BUILD-HOST-CHECKPOINT-ASAN-'+str(ns),cmd('g++','-std=c++17','-g','-O1','-fno-omit-frame-pointer','-fsanitize=address,undefined','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(ROOT/'applications/CHMT'),src,ROOT/'applications/CHMT/restart/Checkpoint.C','-o',binary),[5,19,26,28],900,'Actualproduction checkpointcodec withASan/UBSan; hostreadingrealnativeGPUoutputs only, noGPUmock.',inputs=[src,ROOT/'applications/CHMT/restart/Checkpoint.C'],kind='BUILD')
 if t['exit_code']:continue
 for name in names:
  folder=D/'artifacts/extra'/name
  for checkpoint in sorted(folder.glob('checkpoint-*')):
   t=run('HOST-DECODE-'+name+'-'+checkpoint.name,cmd(binary,checkpoint),[5,8,9,10,14,19,22,26,28],expected='Originalproductionreader exactactualGPUcheckpoint fields withASan/UBSan; allnativephysicalstage/audit/budgetdata retained.',binary=binary,inputs=[checkpoint/'state.chmt'],kind='HOST_SANITIZER')
   raw=(D/t['log']).read_text();ok=False
   if not t['exit_code']:
    try:a=json.loads(raw);(folder/(checkpoint.name+'-decoded.json')).write_text(json.dumps(a,indent=2)+'\n');ok=True
    except ValueError:pass
   assertion(t,ok,'Host ASan/UBSan on actualproductioncodec and nativeGPUcheckpoint; decodedfields JSON '+str(ok))
source=D/'inputs/actual-chemistry-volume-observer.cpp';shutil.copy2(P/'r2_volume_observer.cpp',source);lib=ROOT/'work/actual-chemistry-volume-observer.so'
t=run('BUILD-ACTUAL-CHEM-VOLUME-OBSERVER',cmd('g++','-std=c++17','-O2','-shared','-fPIC','-I/usr/local/cuda/include',source,'-L/usr/local/cuda/lib64','-lcudart','-ldl','-o',lib),[8,10,14,26,29],expected='Externalruntimekernel-argument observer only: actualCUDA funcname and actualD2H8validcellvolumevalues; no sourcearithmetic change.',inputs=[source],kind='BUILD',variant='diagnostic-volume')
if not t['exit_code']:
 original=ROOT/'work/cases/extra/moving-chemistry';case=ROOT/'work/volume-cases/moving-chemistry';shutil.copytree(D/'inputs/native/extra/moving-chemistry',case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(original/rel,case/rel)
 b=ROOT/'work/install/ns10/bin/CHMT'
 t=run('ACTUAL-MOVING-CHEM-VOLUMES',cmd('env','LD_PRELOAD='+str(lib),b,'-case',case),[8,10,14,26,29],expected='ActualNs10kernelarguments and8cellold/new volumes perchemhalf; existingmovingchemistryCSV/checkpoint/geometry allbytesunchanged. ExtraD2H/IO diagnostic excludedfromtiming.',inputs=[source,*[p for p in (D/'inputs/native/extra/moving-chemistry').rglob('*') if p.is_file()]],binary=b,variant='diagnostic-volume',kind='DIAGNOSTIC')
 cmp=compare_dirs(original/'chmtOutput',case/'chmtOutput');(D/'artifacts/extra/moving-chemistry/volume-observer-output-comparison.json').write_text(json.dumps(cmp,indent=2)+'\n')
 raw=(D/t['log']).read_text();rows=[x.split(',') for x in raw.splitlines() if x.startswith('CHEM_VOLUME,')];values=[{'sequence':int(x[1]),'dt_hex':x[2],'native_refinement':int(x[3]),'pointer':x[4],'copy_code':int(x[5]),'volume_hex':x[6:]} for x in rows]
 (D/'artifacts/extra/moving-chemistry/actual-chemical-kernel-volume-inputs.json').write_text(json.dumps(values,indent=2)+'\n')
 assertion(t,cmp['all_files_byte_equal'] and bool(values) and all(x['copy_code']==0 for x in values),'Actual '+str(len(values))+'chemical launches volumeinputs recorded; alloriginalphysicaloutputs bytes unchanged '+str(cmp['all_files_byte_equal']))
