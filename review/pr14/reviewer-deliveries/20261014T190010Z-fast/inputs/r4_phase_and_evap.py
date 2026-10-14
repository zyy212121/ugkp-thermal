from r4_runner import *
import csv
b=ROOT/'work/install/ns2/bin/CHMT'
for mode in ['melting','solidification']:
 oldcase=OLDROOT/'work/cases/phase'/('onecell-'+mode+'-S308');case=ROOT/'work/cases'/('phase-'+mode);assert not case.exists();case.mkdir(parents=True)
 for n in ['0','constant','system']:shutil.copytree(oldcase/n,case/n)
 inp=D/'inputs/cases'/case.name;shutil.copytree(case,inp)
 q=run('GPU-PHASE-'+mode,cmd(b,'-case',case),[3,6,13,14,20,21,26,29],expected='Originallegalnonzero-phaseS308input/mesh/FP64/1e-6 endpoint, actualnativeGPU+OFmaterialowner; signedphasepacket andphysicalold-referencecomparisonseparate.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b,progress_dirs=[case/'chmtOutput'])
 out=archive_output(case,case.name);raw=(D/q['log']).read_text();rows=[]
 for p in sorted((case/'chmtOutput').glob('accepted-*.csv'),key=lambda p:int(p.stem.split('-')[-1])):
  with p.open(newline='') as f:rows.extend(csv.DictReader(f))
 ok=q['exit_code']==0 and rows and float(rows[-1]['time'])==1e-6
 comparisons=[]
 for prefix in ['gas','material','exchange']:
  for p in sorted((case/'chmtOutput').glob(prefix+'-*.csv')):
   op=oldcase/'chmtOutput'/p.name
   comparisons.append({'path':p.name,'old_sha256':sha(op) if op.is_file() else None,'new_sha256':sha(p),'byte_equal':op.is_file() and p.read_bytes()==op.read_bytes()})
 record=D/'artifacts'/(case.name+'-physical-old-source-comparison.json');record.write_text(json.dumps({'source_commit':SOURCE,'old_source_commit':PATHS['old_source'],'original_input_unchanged':True,'final_accepted':rows[-1] if rows else None,'physical_csv_comparisons':comparisons,'all_physical_csv_equal':all(a['byte_equal'] for a in comparisons),'checkpoint_raw_budget_support_compatibility_not_physical_pass':True},indent=2)+'\n')
 assertion(q,ok and bool(comparisons) and all(a['byte_equal'] for a in comparisons),'Actualnative legalphase completed andgas/material/exchange physicalCSV exactoldsource='+str(all(a['byte_equal'] for a in comparisons))+'; fullinventory/SST/target observer supplementary.',artifacts=[str(out.relative_to(D)),str(record.relative_to(D))])
# Prerequisite native film components and real material-consumer cases must have executed.
tests={q['id']:q for q in json.loads((D/'results.json').read_text())['tests']}
assert all(tests['HOST-NATIVE-'+n]['verdict']=='PASS' for n in ['film','curved','lifecycle'])
case=ROOT/'work/cases/wet-evaporation';inputs=[p for p in (D/'inputs/cases/wet-evaporation').rglob('*') if p.is_file()]
q=run('GPU-ORIGINAL-EVAP-ENDPOINT',cmd(b,'-case',case),[3,6,13,14,20,21,26,29],expected='Originalevapinput/physicalconfiguration/mesh/full.001 endpoint; noautomatictime termination, disablingevaporation, endshorteningornew numericalgate.',inputs=inputs,binary=b,estimate_seconds=900,progress_dirs=[case/'chmtOutput'])
out=archive_output(case,'wet-evaporation');rows=[]
for p in sorted((case/'chmtOutput').glob('accepted-*.csv'),key=lambda p:int(p.stem.split('-')[-1])):
 with p.open(newline='') as f:rows.extend(csv.DictReader(f))
ok=q['exit_code']==0 and bool(rows) and float(rows[-1]['time'])==.001
amend(q['id'],verdict='INCONCLUSIVE' if ok else 'FAIL',observations='Originalendpointcompleted='+str(ok)+'; independentfullphysicalstock/flux acceptance separate. Old900/.00025/tinydt/SIGFPE histories retained.',artifacts=[str(out.relative_to(D))],final_accepted=rows[-1] if rows else None)
print('EVAP_ORIGINAL_ENDPOINT',ok,rows[-1] if rows else None,flush=True)
