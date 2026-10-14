from r3_runner import *
import re,collections
native=ROOT/'work/install/ns2/bin/CHMT';original=OLDROOT/'work/install/ns2/bin/CHMT';observer=ROOT/'work/fixtures/CHMT-phase-ledger-observer-D302'
def copycase(base,name):
 case=base.parent/name;assert not case.exists();shutil.copytree(base,case,ignore=shutil.ignore_patterns('chmtOutput'));return case
def inputs(case):return [p for p in case.rglob('*') if p.is_file() and 'chmtOutput' not in p.parts]
def compare(a,b):
 names={str(p.relative_to(a)) for p in a.rglob('*') if p.is_file()}|{str(p.relative_to(b)) for p in b.rglob('*') if p.is_file()};rows=[]
 for n in sorted(names):
  x=a/n;y=b/n;rows.append({'path':n,'left_sha256':sha(x) if x.is_file() else None,'right_sha256':sha(y) if y.is_file() else None,'byte_equal':x.is_file() and y.is_file() and x.read_bytes()==y.read_bytes()})
 return {'all_files_byte_equal':bool(rows) and all(q['byte_equal'] for q in rows),'files':rows}
reports={}
for name in ['melting-positive','solidification-negative']:
 base=ROOT/'work/cases/phase'/name;assert base.is_dir()
 t=run('PHASE-FORMAL-'+name,cmd(native,'-case',base),[3,6,13,14,20,21,26,29],300,'Validselectedmaterial0/nonzero signedphase/newsmallinput; nativeformal9d, physicalstockaudits separately.',inputs=inputs(base),binary=native);archive_output(base,t['id'])
 control=copycase(base,name+'-old4b')
 u=run('PHASE-OLD4B-'+name,cmd(original,'-case',control),[3,6,13,14,20,21,26,29],300,'Safelegal selectedmaterial0 sameALLinput/flags beforefix; no invalidindex/oldOOBcase.',inputs=inputs(control),binary=original,source=PATHS['old_source'],variant='ORIGINAL_4B_SAFE_CONTROL');archive_output(control,u['id'])
 diag=copycase(base,name+'-observer')
 v=run('PHASE-OBSERVER-'+name,cmd(observer,'-case',diag),[3,6,13,14,20,21,26,29],300,'Readonlyphase/finalledger nativeobserver; originals allbytesunchanged required.',inputs=inputs(diag)+[D/'inputs/CpuMaterialDriver-phase-observer-D302.C'],binary=observer,variant='DIAGNOSTIC_D302');archive_output(diag,v['id'])
 prior=compare(base/'chmtOutput',control/'chmtOutput');observed=compare(base/'chmtOutput',diag/'chmtOutput');raw=(D/v['log']).read_text();lines=[q for q in raw.splitlines() if q.startswith(('R3PHASE','R3CONFIG'))];signs=[float.fromhex(q.split('mass=',1)[1].split(',',1)[0]) for q in lines if q.startswith('R3PHASE')]
 sign_ok=bool(signs) and (any(x>0 for x in signs) if name.startswith('melting') else any(x<0 for x in signs))
 report={'same_before_after_outputs':prior,'observer_outputs':observed,'phase_records':lines,'actual_signed_mass_min':min(signs) if signs else None,'actual_signed_mass_max':max(signs) if signs else None,'requested_direction_observed':sign_ok,'source':SOURCE,'original_source':PATHS['old_source'],'scope':'Outputequivalence andactualnonzerobranch/control; fullindependentM/E/P budgetseparate.'};reports[name]=report
 p=D/'artifacts'/('phase-'+name+'-comparison.json');p.write_text(json.dumps(report,indent=2)+'\n')
 assertion(t,t['exit_code']==0 and u['exit_code']==0 and prior['all_files_byte_equal'] and sign_ok,'Originalsafesuccessalloutputsbitexact='+str(prior['all_files_byte_equal'])+', actualnonzerosigndirection='+str(sign_ok)+'; fullstockscope separate.',artifacts=[str(p.relative_to(D))])
 assertion(u,u['exit_code']==0 and prior['all_files_byte_equal'],'Beforefixlegalcontrol actualalloutputbytes comparison='+str(prior['all_files_byte_equal']))
 assertion(v,v['exit_code']==0 and observed['all_files_byte_equal'] and sign_ok,'Readonlyobserverallnativeoutputbytes preserved='+str(observed['all_files_byte_equal'])+', consumedphasecontrol/properties/signedmass observed='+str(sign_ok))
 print(name,'sign',min(signs) if signs else None,max(signs) if signs else None,'priorbytes',prior['all_files_byte_equal'],'observerbytes',observed['all_files_byte_equal'],flush=True)
base=ROOT/'work/cases/extra/wet-film-formal';case=copycase(base,'wet-film-formal-observer')
t=run('WET-FILM-OBSERVER-D302',cmd(observer,'-case',case),[3,6,13,14,20,21,26,29],300,'SameoriginalnoMelting0.001input,actualphaseheat/forceforcing/finalweightedgasfaceflux; observermustalloutputsbytepreserve.',inputs=inputs(case)+[D/'inputs/CpuMaterialDriver-phase-observer-D302.C'],binary=observer,variant='DIAGNOSTIC_D302');archive_output(case,t['id']);cmp=compare(base/'chmtOutput',case/'chmtOutput');(D/'artifacts/wet-observer-output-comparison.json').write_text(json.dumps(cmp,indent=2)+'\n');assertion(t,t['exit_code']==0 and cmp['all_files_byte_equal'],'Actualformalwetalloutputbytes observerpreserved='+str(cmp['all_files_byte_equal'])+'; fullpacket/flux/phasephysicaldata captured.',artifacts=['artifacts/wet-observer-output-comparison.json'])
