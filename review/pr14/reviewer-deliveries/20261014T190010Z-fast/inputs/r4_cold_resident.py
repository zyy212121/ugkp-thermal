from r4_runner import *
import importlib.util,numpy as np,sys
helper_source=D/'inputs/resident-original-caloric-helpers.py';helper_source.write_text(old_read('inputs/resident-original-caloric-helpers-S306.py').decode().replace('from r3_runner import ROOT,D','from r4_runner import ROOT,D').replace('from r2_runner import ROOT,D','from r4_runner import ROOT,D'))
sp=importlib.util.spec_from_file_location('r4_actual_particle_decoder',helper_source);helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
for app in (['FSH'] if '--analysis-only' in sys.argv else ['CHT','FSH']):
 name=app+'-active16-ASCII-S306';case=ROOT/'work/cases'/name;b=ROOT/'work/install/ns2/bin'/app
 if '--analysis-only' in sys.argv:t=next(q for q in json.loads((D/'results.json').read_text())['tests'] if q['id']=='GPU-'+name)
 else:t=run('GPU-'+name,cmd(b,'-case',case),[12,18,19,20,23,26,33],expected='Exactoriginal16activeASCII physicalcold2Dinput, finalnativeO3binaryandnewsharedwire/storage; doublemean+carry original1e-9K alltimes, actualprofilegroups.',inputs=[p for p in (D/'inputs/cases'/name).rglob('*') if p.is_file()],binary=b)
 folder=D/'artifacts/resident'/name;folder.mkdir(parents=True,exist_ok=True);points=[];comparisons=[]
 for q in sorted(case.iterdir()):
  try:stamp=float(q.name)
  except ValueError:continue
  if not q.is_dir() or stamp==0:continue
  shutil.copytree(q,folder/q.name,dirs_exist_ok=True)
  prior=OLDROOT/'work/cases/resident'/name/q.name
  for p in q.rglob('*'):
   if p.is_file() and (prior/p.relative_to(q)).is_file():comparisons.append({'path':str(p.relative_to(case)),'old_sha256':sha(prior/p.relative_to(q)),'new_sha256':sha(p),'byte_equal':p.read_bytes()==(prior/p.relative_to(q)).read_bytes()})
  state=helper.particles(q/'gpuResidentStrictParticles.dat');n=len(state['pm']);nodes=state['pCold2DNodeSpecificEnthalpy'].reshape(n,-1);carry_present='pThermalEnthalpyResidual' in state;carry=state.get('pThermalEnthalpyResidual',np.zeros(n));active=state['pStuck']!=0
  # Original FSH64 schema/rule has no float-cast residual field; this is not an audit/reserved zero.
  nonfinite=[k for k,v in state.items() if np.issubdtype(v.dtype,np.floating) and not np.isfinite(v).all()]
  for i in range(n):
   mean=sum(map(float,nodes[i]))/nodes.shape[1];temp=helper.caloric(mean+float(carry[i]));err=abs(float(state['pT'][i])-temp)
   points.append({'time':q.name,'particle':i,'actual_particle_count':n,'active_stuck':bool(active[i]),'profile_dtype':str(nodes.dtype),'double_mean':mean,'double_carry':float(carry[i]),'carry_field_present':carry_present,'carry_rule':'actualserializedphysicalfield' if carry_present else 'originalFSH64profile_rule_no_float_cast_carry_storage','nonfinite_serialized_fields':nonfinite,'profile_inverse_T':temp,'actual_pT':float(state['pT'][i]),'absolute_error_K':err,'age_progressed':bool(np.any(state['pCold2DRingContactAge'].reshape(n,-1)[i]>0))})
 ok=t['exit_code']==0 and points and all(q['actual_particle_count']==16 and q['active_stuck'] and not q['nonfinite_serialized_fields'] and np.isfinite(q['absolute_error_K']) and q['absolute_error_K']<=1e-9 and (app!='FSH' or q['profile_dtype']=='float64') for q in points) and all(q['age_progressed'] for q in points if float(q['time'])==max(float(x['time']) for x in points))
 artifact=D/'artifacts/resident'/(name+'-actual-profile.json');artifact.write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'original_caloric_rule':'unchangedoriginal analytic Cp/phase inverse','actual_rows':points,'original_absolute_K_threshold':1e-9,'maximum_error_K':max(q['absolute_error_K'] for q in points) if points else None,'physical_scope_pass':bool(ok),'all_corresponding_old_native_files_byte_equal':all(q['byte_equal'] for q in comparisons),'old_source':PATHS['old_source'],'file_comparisons':comparisons,'limits':'cold2Dactiveprofiles only; no wholepairedgas/wall/radiation/M-P-E/species assertion from profiles or finite outputs.'},indent=2)+'\n')
 assertion(t,bool(ok),'Actual16activeprofiles alltimes doublemean+carry max='+str(json.loads(artifact.read_text())['maximum_error_K'])+'K; original1e-9, originaloutput bytes comparatorseparate.',artifacts=[str(artifact.relative_to(D)),str(folder.relative_to(D))]);print('COLD_ACTUAL',app,ok,json.loads(artifact.read_text())['maximum_error_K'],flush=True)
