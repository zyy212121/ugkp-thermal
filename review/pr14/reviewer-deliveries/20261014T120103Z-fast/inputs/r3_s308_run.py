from r3_runner import *
import json
def inputs(c):return [q for q in c.rglob('*') if q.is_file() and 'chmtOutput' not in q.parts]
def compare(a,b):
 names={str(p.relative_to(a)) for p in a.rglob('*') if p.is_file()}|{str(p.relative_to(b)) for p in b.rglob('*') if p.is_file()}
 rows=[{'path':n,'left_sha256':sha(a/n) if (a/n).is_file() else None,'right_sha256':sha(b/n) if (b/n).is_file() else None,'byte_equal':(a/n).is_file() and (b/n).is_file() and (a/n).read_bytes()==(b/n).read_bytes()} for n in sorted(names)]
 return {'all_files_byte_equal':bool(rows) and all(q['byte_equal'] for q in rows),'files':rows}
for name in ['onecell-melting-S308','onecell-solidification-S308']:
 case=ROOT/'work/cases/phase'/name;b=ROOT/'work/install/ns2/bin/CHMT';t=run('PHASE-FORMAL-'+name,cmd(b,'-case',case),[3,6,13,14,20,21,26,29],300,'Originaldeclared legalonecellhost-equivalent controlsupportgeometry corrected; no numericalproperty/end change.',inputs=inputs(case),binary=b,variant='SUPPORT_INPUT_S308_FORMAL_NATIVE');archive_output(case,t['id'])
 if t['exit_code']!=0:
  print('S308_NONCOMPLETE',name,'no furtherinput/budget changes',flush=True);continue
 old=case.parent/(name+'-old4b');shutil.copytree(case,old,ignore=shutil.ignore_patterns('chmtOutput'));b0=OLDROOT/'work/install/ns2/bin/CHMT';u=run('PHASE-OLD4B-'+name,cmd(b0,'-case',old),[3,6,13,14,20,21,26,29],300,'Safelegalselectedmaterial0 sameonecellinput/flags beforefix.',inputs=inputs(old),binary=b0,source=PATHS['old_source'],variant='ORIGINAL_4B_SAFE_CONTROL');archive_output(old,u['id']);cmp=compare(case/'chmtOutput',old/'chmtOutput')
 diag=case.parent/(name+'-D302');shutil.copytree(case,diag,ignore=shutil.ignore_patterns('chmtOutput'));bd=ROOT/'work/fixtures/CHMT-phase-ledger-observer-D302';v=run('PHASE-D302-'+name,cmd(bd,'-case',diag),[3,6,13,14,20,21,26,29],300,'Read-only actualfullsignedphase/volume/forcing/stock/weight chain.',inputs=inputs(diag),binary=bd,variant='DIAGNOSTIC_D302');archive_output(diag,v['id']);obs=compare(case/'chmtOutput',diag/'chmtOutput');p=D/'artifacts'/('phase-'+name+'-before-after.json');p.write_text(json.dumps({'formal_before_after':cmp,'observer_compare':obs},indent=2)+'\n');assertion(t,u['exit_code']==0 and cmp['all_files_byte_equal'],'Nativefulllegal controlbefore/afterrawparity '+str(cmp['all_files_byte_equal'])+'; checkpointproduceridentity exceptions/fullstock separate',artifacts=[str(p.relative_to(D))]);assertion(u,cmp['all_files_byte_equal'],'Safeoldcompletebitcomparison');assertion(v,obs['all_files_byte_equal'],'Readonlyalloutputbytes='+str(obs['all_files_byte_equal']))
