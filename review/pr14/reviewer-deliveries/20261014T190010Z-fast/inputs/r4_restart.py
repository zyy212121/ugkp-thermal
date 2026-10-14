from r4_runner import *
import csv,struct
b=ROOT/'work/install/ns2/bin/CHMT'
for name in ['thermal-full','wet-film']:
 full=ROOT/'work/cases'/name;case=ROOT/'work/cases'/(name+'-restart');assert not case.exists();case.mkdir()
 for folder in ['0','constant','system']:shutil.copytree(full/folder,case/folder)
 cp=full/'chmtOutput/checkpoint-1';assert cp.is_dir()
 prop=case/'constant/chmtProperties';prop.write_text(prop.read_text()+'\nrestartDirectory "'+str(cp)+'";\n')
 inp=D/'inputs/cases'/case.name;shutil.copytree(case,inp)
 t=run('GPU-RESTART-'+name,cmd(b,'-case',case),[5,6,13,14,19,26,29],expected='Originalfull checkpoint1 andsame end.001/physicalflags; shortestaffectedformal continuation; strictphysicalbits no cache/coldseed/extraiteration edit.',inputs=[*[p for p in inp.rglob('*') if p.is_file()],cp/'state.chmt'],binary=b)
 out=archive_output(case,case.name);differences=[];comparisons=[]
 for p in sorted((case/'chmtOutput').glob('*.csv')):
  ref=full/'chmtOutput'/p.name
  if not ref.is_file():continue
  comparisons.append({'file':p.name,'uninterrupted_sha256':sha(ref),'restart_sha256':sha(p),'byte_equal':p.read_bytes()==ref.read_bytes()})
  if p.name.startswith(('gas-','material-','exchange-')):
   with p.open(newline='') as f:a=list(csv.DictReader(f))
   with ref.open(newline='') as f:c=list(csv.DictReader(f))
   for row,(left,right) in enumerate(zip(c,a)):
    for key in left:
     if left[key]!=right[key]:
      try:x,y=float(left[key]),float(right[key]);same=struct.pack('<d',x)==struct.pack('<d',y)
      except ValueError:same=left[key]==right[key];x,y=left[key],right[key]
      if not same:differences.append({'file':p.name,'row':row,'field':key,'uninterrupted':left[key],'restart':right[key],'uninterrupted_hex':x.hex() if isinstance(x,float) else None,'restart_hex':y.hex() if isinstance(y,float) else None})
 initial=case/'chmtOutput/checkpoint-1/state.chmt';initial_equal=initial.is_file() and initial.read_bytes()==(cp/'state.chmt').read_bytes()
 summary={'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'checkpoint_input_sha256':sha(cp/'state.chmt'),'initial_reserialized_checkpoint_bytes_equal':initial_equal,'all_csv_comparisons':comparisons,'strict_physical_semantic_bit_difference_count':len(differences),'first_difference':differences[0] if differences else None,'all_differences':differences,'no_automatic_numerical_approval':True,'legacy_reserved_slots_not_measured':True}
 p=D/'artifacts'/(name+'-strict-restart.json');p.write_text(json.dumps(summary,indent=2)+'\n')
 assertion(t,t['exit_code']==0 and initial_equal and not differences,'Actualformalstrict restart physicaldifferencecount='+str(len(differences))+' first='+json.dumps(summary['first_difference'])+'; no smallerror/tolerance/retiredslot exemption.',artifacts=[str(out.relative_to(D)),str(p.relative_to(D))])
 print('RESTART_STRICT',name,len(differences),summary['first_difference'],flush=True)
