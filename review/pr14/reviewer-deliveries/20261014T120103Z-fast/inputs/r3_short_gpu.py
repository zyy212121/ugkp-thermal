from r3_runner import *
import re,collections
def inp(case):return [p for p in case.rglob('*') if p.is_file() and 'chmtOutput' not in p.parts and p.name!='gpuResidentStrict.log']
def compare(a,b):
 ns={str(p.relative_to(a)) for p in a.rglob('*') if p.is_file()}|{str(p.relative_to(b)) for p in b.rglob('*') if p.is_file()};rows=[]
 for n in sorted(ns):
  x=a/n;y=b/n;rows.append({'path':n,'left_sha256':sha(x) if x.is_file() else None,'right_sha256':sha(y) if y.is_file() else None,'byte_equal':x.is_file() and y.is_file() and x.read_bytes()==y.read_bytes()})
 return {'all_files_byte_equal':bool(rows) and all(q['byte_equal'] for q in rows),'files':rows}
b=ROOT/'work/fixtures/chemistry-owner-entry-S305';t=run('CHEMISTRY-OWNER-ENTRY-S305',cmd(b),[10,12,14,26],300,'Actual advanceGasTrial dt/time/native atentry chronology +alloriginalpositive semanticassertions.',inputs=[D/'inputs/GpuGasAdvance-owner-entry-S305.cuh',D/'inputs/chemistry-owner-chronology-S304.cu'],binary=b,variant='DIAGNOSTIC_S305')
raw=(D/t['log']).read_text();original=(D/'logs/CHEMISTRY-FORMAL-FIXTURE.log').read_text().split('PASS failed-cell')[0].strip();filtered='\n'.join(q for q in raw.splitlines() if not q.startswith(('R3TRIAL','R3CALL','PASS real CUDA'))).strip();rows=[q for q in raw.splitlines() if q.startswith(('R3TRIAL','R3CALL'))]
calls=[{'dt_hex':a,'time_hex':c,'native':int(n)} for a,c,n in re.findall(r'R3CALL dt=(\S+) time=(\S+) native=(\d+)',raw)]
first=raw.split('PASS requested owner accepted trace')[0];firstcalls=[float.fromhex(a) for a in re.findall(r'R3CALL dt=(\S+)',first)];ok=t['exit_code']==0 and filtered==original and firstcalls[:3]==[4e-6,2e-6,1e-6] and 'event=rollback attempt=1' in first and 'stepLimits=1 retry=1' in first
p=D/'artifacts/chemistry-owner-entry-S305.json';p.write_text(json.dumps({'all_original_owner_semantic_stdout_bitexact':filtered==original,'actual_native_entry_calls':calls,'raw_chronology':rows,'first_attempts':firstcalls,'first_requested_4e6_StepLimit_then_halves':ok,'source':SOURCE,'S304_correction':'Ignore currentDuration dt in rejected R3TRIAL rows; R3CALL records actualdt, existing begin ids/order link the calls.'},indent=2)+'\n');assertion(t,ok,'Nativeentryactual4e-6→2e-6→1e-6 firstStepLimitrollback +alloriginalsemanticownerstdout bitexact='+str(filtered==original),artifacts=[str(p.relative_to(D))]);print('\n'.join(rows[:40]),flush=True)
for app in ['CHT','FSH']:
 case=ROOT/('work/cases/resident/'+app+'-active16-ASCII-S306');b=ROOT/'work/install/ns2/bin'/app
 t=run('NATIVE-'+case.name,cmd(b,'-case',case),[5,12,18,19,20,23,26,33],300,'ActualauthoritativeASCII mobileingress16contactedparcels, twoactivecompletegroups, originalpT/profile mean+carry1e-9; auditedseparately.',inputs=inp(case),binary=b,source=PATHS['old_source'],variant='UNCHANGED_DEPENDENCY_REUSED_NATIVE_BINARY')
 out=D/'artifacts/resident'/case.name;out.mkdir(parents=True,exist_ok=True)
 for p in case.iterdir():
  try:float(p.name)
  except ValueError:continue
  if p.is_dir():shutil.copytree(p,out/p.name)
 if (case/'gpuResidentStrict.log').exists():shutil.copy2(case/'gpuResidentStrict.log',out/'gpuResidentStrict.log')
 amend(t['id'],observations='Actualnativeexit recorded; activecount/contact/age/profile/launch witness andpairedbudgets pending.',artifacts=[str(out.relative_to(D))])
for name,oldname in [('thermal-full','thermal-full'),('thermal-restart','thermal-formal-resume')]:
 case=ROOT/'work/cases/chmt'/name;b=ROOT/'work/fixtures/CHMT-thermal-seed-D303';t=run('THERMAL-SEED-'+name,cmd(b,'-case',case),[5,9,12,14,26],300,'ReadonlyTgas seed currentCo/selecteddt/actualfirstfinalflux packet; originalfull/actualcheckpoint1restartexactinput.',inputs=inp(case),binary=b,variant='DIAGNOSTIC_D303');archive_output(case,t['id'])
 cmp=compare(OLDROOT/'work/cases/chmt'/oldname/'chmtOutput',case/'chmtOutput');p=D/'artifacts'/('thermal-seed-output-'+name+'.json');p.write_text(json.dumps(cmp,indent=2)+'\n');assertion(t,t['exit_code']==0 and cmp['all_files_byte_equal'],'Readonlyobserveroutputs compared to correspondingoriginalformal4b '+str(cmp['all_files_byte_equal'])+'; sourceidentity metadata exceptions separately quantified ifneeded.',artifacts=[str(p.relative_to(D))])
for name in ['onecell-melting','onecell-solidification']:
 case=ROOT/'work/cases/phase'/name;b=ROOT/'work/install/ns2/bin/CHMT';t=run('PHASE-FORMAL-'+name,cmd(b,'-case',case),[3,6,13,14,20,21,26,29],300,'Independentexistinglegalhost-probe equivalentonecell actualnativeGPU; nonzero actualphase and fullMEP before/after stillrequired.',inputs=inp(case),binary=b);archive_output(case,t['id'])
 if t['exit_code']!=0:
  print('ONECELL_NOTCOMPLETE',name,'No redundant before/after or observer repetition after incompleteformalcontrol; actualnonzero8cellpartial path retained.',flush=True);continue
 old=case.parent/(name+'-old4b');shutil.copytree(case,old,ignore=shutil.ignore_patterns('chmtOutput'));b0=OLDROOT/'work/install/ns2/bin/CHMT';u=run('PHASE-OLD4B-'+name,cmd(b0,'-case',old),[3,6,13,14,20,21,26,29],300,'Safelegalselectedphase0 samecompleteonecell input/flags beforefix.',inputs=inp(old),binary=b0,source=PATHS['old_source'],variant='ORIGINAL_4B_SAFE_CONTROL');archive_output(old,u['id']);cmp=compare(case/'chmtOutput',old/'chmtOutput')
 observer=case.parent/(name+'-D302');shutil.copytree(case,observer,ignore=shutil.ignore_patterns('chmtOutput'));bd=ROOT/'work/fixtures/CHMT-phase-ledger-observer-D302';v=run('PHASE-D302-'+name,cmd(bd,'-case',observer),[3,6,13,14,20,21,26,29],300,'Read-onlyactualnonzerophase/force/volume finalledger; nooutputchanges.',inputs=inp(observer),binary=bd,variant='DIAGNOSTIC_D302');archive_output(observer,v['id']);obscmp=compare(case/'chmtOutput',observer/'chmtOutput')
 p=D/'artifacts'/('phase-'+name+'-before-after.json');p.write_text(json.dumps({'formal_before_after':cmp,'observer_compare':obscmp},indent=2)+'\n');assertion(t,u['exit_code']==0 and cmp['all_files_byte_equal'],'Actuallegalbefore/after fullraw bitparity='+str(cmp['all_files_byte_equal'])+'; branch/fullstock auditseparate',artifacts=[str(p.relative_to(D))]);assertion(u,cmp['all_files_byte_equal'],'Safeoldlegal alloutputs exact comparison');assertion(v,obscmp['all_files_byte_equal'],'Observerrawoutput equality='+str(obscmp['all_files_byte_equal']))
