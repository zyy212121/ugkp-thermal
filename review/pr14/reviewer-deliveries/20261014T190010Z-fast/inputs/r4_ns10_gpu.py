from r4_runner import *
import csv
b=ROOT/'work/install/ns10/bin/CHMT'
for name,sanitizer in [('chemistry-full',True),('moving-chemistry',False)]:
 case=ROOT/'work/cases'/name;inp=[p for p in (D/'inputs/cases'/name).rglob('*') if p.is_file()]
 args=cmd('compute-sanitizer','--tool','initcheck','--error-exitcode','99',b,'-case',case) if sanitizer else cmd(b,'-case',case)
 t=run('GPU-Ns10-'+name,args,[8,9,10,14,24,26,28,29],expected='Exactoriginal Ns10 input/end2e-5, finalsharedsource470+originalO2build, physicalassertionsseparate; initcheck requirescompletedERRORSUMMARY0.',inputs=inp,binary=b,kind='SANITIZER' if sanitizer else 'GPU',estimate_seconds=600,progress_dirs=[case/'chmtOutput'])
 out=archive_output(case,'Ns10-'+name);q=sorted((case/'chmtOutput').glob('accepted-*.csv'),key=lambda p:int(p.stem.split('-')[-1]));rows=[]
 for p in q:
  with p.open(newline='') as f:rows.extend(csv.DictReader(f))
 raw=(D/t['log']).read_text();ok=t['exit_code']==0 and bool(rows) and float(rows[-1]['time'])==2e-5
 if sanitizer:ok=ok and 'ERROR SUMMARY: 0 errors' in raw
 amend(t['id'],verdict='INCONCLUSIVE' if ok else 'FAIL',observations='Nativewholeoriginalendpointcompleted='+str(ok)+'; flux/volume/physicalinventory andallassertionsscope separate.',artifacts=[str(out.relative_to(D))],final_accepted=rows[-1] if rows else None)
 if ok and sanitizer:
  checker=D/'inputs/check_case_metadata_support.py'
  a=run('CHECK-S401-Ns10-chemistry-full',cmd('python3',checker,case),[8,9,10,14,24,26,28,29],expected='Originalphysicalchemical/EOS/species/element/contact/checkpoint/endrules; onlyNOT_COLLECTEDmetadatacolumnrecognized byS401.',inputs=[checker,case/'verification.json'],kind='ANALYSIS',variant='SUPPORT_S401')
  assertion(a,a['exit_code']==0,(D/a['log']).read_text(errors='replace'))
