from r4_runner import *
import csv,struct
bin2=ROOT/'work/install/ns2/bin/CHMT'
for name,sanitizer in [('thermal-full','initcheck'),('wet-film',None),('SST-RK',None),('SST-SSPRK2',None),('SST-SSPRK3',None),('moving-GCL',None)]:
 case=ROOT/'work/cases'/name
 if not case.is_dir():continue
 inputs=[p for p in (D/'inputs/cases'/name).rglob('*') if p.is_file()]
 args=cmd('compute-sanitizer','--tool',sanitizer,'--error-exitcode','99',bin2,'-case',case) if sanitizer else cmd(bin2,'-case',case)
 t=run('GPU-'+name,args,[3,6,8,9,13,14,20,21,22,26,29],expected='Originalsmall physicalinput/end/mesh/source470, actualnativeGPU; field/flux/fullinventory assertions separate.',inputs=inputs,binary=bin2,kind='SANITIZER' if sanitizer else 'GPU',estimate_seconds=300,progress_dirs=[case/'chmtOutput'])
 out=archive_output(case,name)
 summaries=sorted((case/'chmtOutput').glob('accepted-*.csv'),key=lambda p:int(p.stem.split('-')[-1]))
 rows=[]
 for p in summaries:
  with p.open(newline='') as f:rows.extend(csv.DictReader(f))
 raw=(D/t['log']).read_text();complete=t['exit_code']==0 and bool(rows) and float(rows[-1]['time'])==.001
 if sanitizer:complete=complete and 'ERROR SUMMARY: 0 errors' in raw
 amend(t['id'],verdict='INCONCLUSIVE' if complete else 'FAIL',observations='Nativeexecution completedoriginalendpoint='+str(complete)+'; independentphysicalfield/flux/inventory acceptance pending, reservedaudit slots notmeasuredzero.',artifacts=[str(out.relative_to(D))],final_accepted=rows[-1] if rows else None)
 print('CASE_RUN',name,'endpoint',rows[-1].get('time') if rows else None,'exit',t['exit_code'],flush=True)
 if name=='thermal-full' and complete:
  checker=ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py'
  q=run('CHECK-ORIGINAL-thermal-full',cmd('python3',checker,case),[6,14,26,29],expected='Originalindependentdry thermalfields/contact/EOS/end checker; no threshold edits.',inputs=[checker,case/'verification.json'],kind='ANALYSIS')
  assertion(q,q['exit_code']==0,'Originalcheckerfullresult '+(D/q['log']).read_text(errors='replace')[-2800:])
