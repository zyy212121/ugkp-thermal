from r4_runner import *
cases=['thermal-full','wet-film','SST-RK','SST-SSPRK2','SST-SSPRK3','moving-GCL','moving-chemistry','chemistry-full','phase-melting','phase-solidification']
for name in cases:
 original=ROOT/'work/cases'/name;case=ROOT/'work/cases'/('D401-'+name);assert not case.exists();case.mkdir(parents=True)
 for n in ['0','constant','system']:shutil.copytree(original/n,case/n)
 ns=10 if 'chemistry' in name else 2;b=ROOT/'work/install'/('diag'+str(ns))/'bin/CHMT'
 t=run('D401-GPU-'+name,cmd(b,'-case',case),[8,9,10,14,20,21,26,29],expected='ReadonlyactualEuler/RK/SST/chemhalf observation withsameformalinput/end; allcorresponding originaloutputbytes mustmatch beforephysicalaudit.',inputs=[p for n in ['0','constant','system'] for p in (case/n).rglob('*') if p.is_file()],binary=b,variant='DIAGNOSTIC_D401')
 rows=[]
 for p in (original/'chmtOutput').rglob('*'):
  if not p.is_file():continue
  q=case/'chmtOutput'/p.relative_to(original/'chmtOutput');rows.append({'path':str(p.relative_to(original/'chmtOutput')),'formal_sha256':sha(p),'observer_sha256':sha(q) if q.is_file() else None,'byte_equal':q.is_file() and p.read_bytes()==q.read_bytes()})
 equal=t['exit_code']==0 and rows and all(q['byte_equal'] for q in rows)
 out=D/'artifacts'/('D401-'+name+'-output-invariance.json');out.write_text(json.dumps({'source_commit':SOURCE,'variant':'DIAGNOSTIC_D401','all_original_files_byte_equal':bool(equal),'comparisons':rows,'observer_input_recipe_same_as_formal':True,'diagnostic_io_excluded_from_efficiency':True},indent=2)+'\n')
 assertion(t,bool(equal),'Readonlyobserveralloriginaloutputfiles byteequal='+str(bool(equal))+' count='+str(len(rows))+'; physicalstage/flux audit separate.',artifacts=[str(out.relative_to(D))])
 print('D401_BYTES',name,equal,len(rows),flush=True)
