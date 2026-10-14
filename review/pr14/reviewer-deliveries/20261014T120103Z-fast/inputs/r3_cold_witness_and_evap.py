from r3_runner import *
import re,threading
def inputs(c):return [p for p in c.rglob('*') if p.is_file() and p.parts[len(c.parts)] in ['0','constant','system']]
for app in ['CHT','FSH']:
 base=ROOT/('work/cases/resident/'+app+'-active16-ASCII-S306');case=base.parent/(app+'-active16-witness-S309');assert not case.exists();case.mkdir()
 for name in ['0','constant','system']:shutil.copytree(base/name,case/name)
 binary=ROOT/'work/install/ns2/bin'/app;lib=ROOT/'work/fixtures/cold-actual-launch-S309.so';t=run('COLD-ACTUAL-LAUNCH-'+app+'-S309',cmd('env','LD_PRELOAD='+str(lib),binary,'-case',case),[12,18,19,20,23,26,33],300,'Sameactual16activeASCII input, actualcoldlaunchdimensionsmetadataonly, alloutputbytescompare.',inputs=inputs(case)+[D/'inputs/cold-actual-kernel-dimensions-S309.cpp'],binary=binary,source=PATHS['old_source'],variant='DIAGNOSTIC_S309')
 rows=[]
 for p in base.rglob('*'):
  if not p.is_file():continue
  try:float(p.relative_to(base).parts[0])
  except ValueError:continue
  q=case/p.relative_to(base);rows.append({'path':str(p.relative_to(base)),'base_sha256':sha(p),'observer_sha256':sha(q) if q.exists() else None,'byte_equal':q.is_file() and p.read_bytes()==q.read_bytes()})
 raw=(D/t['log']).read_text();launches=[q for q in raw.splitlines() if q.startswith('ACTUAL_COLD_LAUNCH')];p=D/'artifacts/resident'/('cold-actual-groups-'+app+'-S309.json');p.write_text(json.dumps({'rawlaunches':launches,'launchcount':len(launches),'allcorrespondingnativeoutputbytes_equal':all(q['byte_equal'] for q in rows),'all_output_comparisons':rows,'actual_active_particles':16,'scope':'Actualgroupkernel dimensions plus16active outputrecords; group size inspected in unchanged native source, no parameter mutation/extraD2H.'},indent=2)+'\n');assertion(t,t['exit_code']==0 and bool(launches) and all(q['byte_equal'] for q in rows),'Actualcoldkernellaunchcount='+str(len(launches))+' allnativeoutputbytes unchanged='+str(all(q['byte_equal'] for q in rows)),artifacts=[str(p.relative_to(D))]);print(app,'LAUNCH',launches[:4],flush=True)
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nD302 原evap900阻断追加原因定位（300s限诊断，不追加完成预算）：原正式900s已acceptedtime.00025，未达.001且无CUDA资源错误；相变8cell观察见非常小localstep，因此对同原evap输入/完整终点做只读phase/finalledger进度定位。每20s外部/proc CPUtime/readwrite统计+nvidia-smi util/memory/logsize与checkpoint时序，不写求解器、不用timing评价效率、不危险gdb attach。期望区分GPU/CPU推进、local accepted候选与macrocommitted，仅部分重现不会算PASS；formal900/block与old300/原SIGFPE永久保留。\n')
base=ROOT/'work/cases/extra/wet-evaporation-formal-900';case=base.parent/'wet-evaporation-progress-D302';shutil.copytree(base,case,ignore=shutil.ignore_patterns('chmtOutput'));binary=ROOT/'work/fixtures/CHMT-phase-ledger-observer-D302';stop=threading.Event();observations=[]
def observe():
 while not stop.wait(20):
  obs={'utc':utc(),'log_bytes':(D/'logs/EVAPORATION-PROGRESS-D302.log').stat().st_size if (D/'logs/EVAPORATION-PROGRESS-D302.log').exists() else 0,'native_processes':[],'checkpoints':[{ 'path':str(p.relative_to(case)), 'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size} for p in (case/'chmtOutput').glob('checkpoint-*/state.chmt')]}
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if command.startswith(str(binary)) and str(case) in command:obs['native_processes'].append({'pid':int(p.name),'stat':(p/'stat').read_text(),'io':(p/'io').read_text(),'status':(p/'status').read_text()})
   except (PermissionError,ProcessLookupError,FileNotFoundError):pass
  result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True,timeout=5);obs['gpu_resource_snapshot']=result.stdout;observations.append(obs);(D/'artifacts/evaporation-progress-resource-timeseries.json').write_text(json.dumps(observations,indent=2)+'\n')
thread=threading.Thread(target=observe,daemon=True);thread.start()
try:t=run('EVAPORATION-PROGRESS-D302',cmd(binary,'-case',case),[3,6,13,14,20,21,26,29],300,'Sameoriginalevap input/end; readonlylocalphase/finalweightedflux andexternalCPU/GPUresourceprogress only. FiniteFASTdiagnosis notcompletionextension.',inputs=inputs(case)+[D/'inputs/CpuMaterialDriver-phase-observer-D302.C'],binary=binary,variant='DIAGNOSTIC_D302');archive_output(case,t['id']);amend(t['id'],observations='Readonlyboundedprogressdiagnosis only; localacceptedphase maynotcommitwholewindow; originalformal900 remainsBLOCKED.',artifacts=['artifacts/evaporation-progress-resource-timeseries.json','artifacts/'+t['id']])
finally:stop.set();thread.join(timeout=6)
