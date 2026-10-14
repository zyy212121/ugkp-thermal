from r3_runner import *
import re,collections
binary=ROOT/'work/fixtures/cold-start-Js-S311';t=run('COLD-START-JS-ACTUAL-S311',cmd(binary),[4,9,12,14,17,26,29],300,'Actualnativeowner/kernel Js/currentCo2flow-dtvariants/freshreferences/laggedpublication actualconsumer; allD2H explicitlysynchronized.',inputs=[D/'inputs/cold-start-Js-actual-native-S310.cu'],binary=binary,variant='SUPPORT_S311_ACTUAL_CUDA')
raw=(D/t['log']).read_text();ok=t['exit_code']==0 and raw.count('R3JS fresh')==6 and raw.count('R3JS lagged')==2 and 'PASS actual CUDA cold-start' in raw;assertion(t,ok,'ActualCUDA6freshreferences/2laggedpublications andoriginalKomega invariant='+str(ok)+'; actualcopyfailure/generalresources notclosed');print(raw[-5000:],flush=True)
for app in ['CHT','FSH']:
 base=ROOT/('work/cases/resident/'+app+'-active16-ASCII-S306');case=base.parent/(app+'-active16-all-launch-S313');assert not case.exists();case.mkdir()
 for name in ['0','constant','system']:shutil.copytree(base/name,case/name)
 binary=ROOT/'work/install/ns2/bin'/app;lib=ROOT/'work/fixtures/all-actual-launch-S313.so';inputs=[p for p in case.rglob('*') if p.is_file()];t=run('ALL-ACTUAL-LAUNCH-'+app+'-S313',cmd('env','LD_PRELOAD='+str(lib),binary,'-case',case),[12,18,19,20,23,26,33],300,'Actualallkernelname/dims unfiltered, same16activecoldphysicalinput, rawalloutputs exactcompare.',inputs=inputs+[D/'inputs/all-actual-kernel-dimensions-S313.cpp'],binary=binary,source=PATHS['old_source'],variant='DIAGNOSTIC_S313')
 rows=[]
 for p in base.rglob('*'):
  if not p.is_file():continue
  try:float(p.relative_to(base).parts[0])
  except ValueError:continue
  q=case/p.relative_to(base);rows.append({'path':str(p.relative_to(base)),'base_sha256':sha(p),'observer_sha256':sha(q) if q.exists() else None,'byte_equal':q.is_file() and p.read_bytes()==q.read_bytes()})
 raw=(D/t['log']).read_text();launches=[q for q in raw.splitlines() if q.startswith('ACTUAL_ALL_LAUNCH')];count=collections.Counter(q.split(',')[1] for q in launches);p=D/'artifacts/resident'/('all-actual-kernel-groups-'+app+'-S313.json');p.write_text(json.dumps({'alllaunchcount':len(launches),'name_counts':dict(count),'all_native_output_bytes_equal':all(q['byte_equal'] for q in rows),'all_output_comparisons':rows,'actual_active_particles':16,'rawlog':t['log']},indent=2)+'\n');assertion(t,t['exit_code']==0 and bool(launches) and all(q['byte_equal'] for q in rows),'Actualunfilteredkernel launch witness; all correspondingnative bytes='+str(all(q['byte_equal'] for q in rows)),artifacts=[str(p.relative_to(D))]);print('ALLKERNEL_THERMAL',app,[(k,v) for k,v in count.items() if 'Stuck' in k or 'Thermal' in k or 'Heat' in k],flush=True)
