from r3_runner import *
import re,csv,decimal,statistics
raw=(D/'logs/EVAPORATION-PROGRESS-D302.log').read_text();phase=[dict(re.findall(r'(\w+)=([^,\s]+)',q)) for q in raw.splitlines() if q.startswith('R3PHASE')];bywindow={}
for row in phase:bywindow.setdefault(row['window'],[]).append(row)
summary={}
for key,rs in bywindow.items():
 dts=[float.fromhex(q['dt']) for q in rs];times=[float.fromhex(q['time']) for q in rs];summary[key]={'phase_rows':len(rs),'fourface_local_candidates':len(rs)//4,'local_dt_min':min(dts),'local_dt_max':max(dts),'first_time':min(times),'last_time':max(times),'first4':rs[:4],'last4':rs[-4:]}
resources=json.loads((D/'artifacts/evaporation-progress-resource-timeseries.json').read_text());cpcompare=[]
formal=D/'artifacts/WET-EVAPORATION-FORMAL-900';diag=D/'artifacts/EVAPORATION-PROGRESS-D302'
for p in formal.rglob('*'):
 if not p.is_file():continue
 q=diag/p.relative_to(formal)
 if q.is_file():cpcompare.append({'path':str(p.relative_to(formal)),'formal_sha256':sha(p),'observer_sha256':sha(q),'byte_equal':p.read_bytes()==q.read_bytes()})
cpu=[]
for obs in resources:
 for proc in obs['native_processes']:
  vals=proc['stat'].split();cpu.append({'utc':obs['utc'],'pid':proc['pid'],'state':vals[2],'utime_ticks':int(vals[13]),'stime_ticks':int(vals[14]),'gpu':obs['gpu_resource_snapshot'],'log_bytes':obs['log_bytes']})
case=ROOT/'work/cases/extra/wet-evaporation-formal-900';accepted=[{'path':str(p.relative_to(case/'chmtOutput')),'raw':p.read_text()} for p in sorted((case/'chmtOutput').glob('accepted-*.csv'))]
report={'identity':'PR14_REVIEW_ONLY','formal_budget_seconds':900,'formal_exit':124,'formal_end_target':.001,'formal_committed_time':.00025,'formal_accepted_raw':accepted,'observer_budget_seconds':300,'observer_exit':124,'local_phase_progress':summary,'resource_timeseries':cpu,'same_completed_observer_vs_formal_files':cpcompare,'all_common_completed_files_byte_equal':all(q['byte_equal'] for q in cpcompare),'observed_CUDA_error_lines':[l for l in raw.splitlines() if any(k in l for k in ['CUDA error','out of memory','copy failed','CHMT failed'])],'classification':'Slow ongoing CPU material/film local subdivision at extremely small dt after firstcommitted .00025; no observedGPUresourcefault, no completed .001 numericalacceptance. Exact internal limiter cause notstack-proven. Local candidates notmacrocompletion, observer IO nottiming.'}
(D/'artifacts/evaporation-blocked-cause-and-progress.json').write_text(json.dumps(report,indent=2)+'\n');print('EVAPPROGRESS',summary.keys(),[(k,q['phase_rows'],q['local_dt_min'],q['last_time']) for k,q in summary.items()],report['all_common_completed_files_byte_equal'],cpu[:1],cpu[-1:])
# Correct derived group count in separate sidecar; original S306 evidence retained.
correction={'identity':'PR14_REVIEW_ONLY','original_jsons_preserved':True,'wrong_original_field':'complete_8lane_groups_from_actual_active_count=2','correct_mapping':'One wall-bound particle consumes 8radiallanes; 16activeparticles/block128 means16 complete8lane heat groups','correct_actual_group_count':16,'source':'common/wall/GpuColdWall2DDevice.cuh','actual_global_wrapper':'relaxColdWall2DParticlesToResidentGasKernel','source_hash':sha(ROOT/'common/wall/GpuColdWall2DDevice.cuh'),'ring_age_note':'Firstoutput initializes contact withzeroages; later ring0 activeage positive for16particles, otherunwet ringageszero are not incomplete execution. DynamicpT/profile andareas actual.','actual_witness':{app:json.loads((D/'artifacts/resident'/('cold-actual-groups-'+app+'-S309.json')).read_text())['allcorrespondingnativeoutputbytes_equal'] for app in ['CHT','FSH']}}
(D/'metadata/cold-group-count-correction.json').write_text(json.dumps(correction,indent=2)+'\n')
# Native build manifests distinguish external exact source pin from optional legacy identity.
info={}
for ns in [2,10]:
 raw=(D/'logs'/('NATIVE-BUILD-IDENTITY-Ns'+str(ns)+'.log')).read_text();info[str(ns)]=json.loads(next(l for l in raw.splitlines() if l.startswith('{')))
(D/'metadata/native-build-identity-parsed.json').write_text(json.dumps(info,indent=2)+'\n');print('IDENTITY_KEYS', {k:{a:b for a,b in v.items() if 'commit' in a or 'source' in a or 'revision' in a or 'precision' in a} for k,v in info.items()})
before=json.loads((D/'versions.json').read_text());before['actual_binaries_objects']={str(p.relative_to(ROOT)):sha(p) for parent in ['work/install','work/fixtures'] for p in (ROOT/parent).rglob('*') if p.is_file() and (os.access(p,os.X_OK) or p.suffix in ['.o','.so'])};before['build_execution_ids']=[q['id'] for q in json.loads((D/'results.json').read_text())['tests'] if q['kind']=='BUILD'];before['native_build_info_ref']='metadata/native-build-identity-parsed.json';(D/'versions.json').write_text(json.dumps(before,indent=2)+'\n')
env={'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','utc':utc(),'host':'Windows11/WSL2 Ubuntu22.04; OpenFOAM10 at/opt/openfoam10','gpu_count':1,'gpu':'NVIDIA GeForce RTX4060 Laptop8GB','driver':'592.00','CUDA':'13.1','gpu_architecture':'sm_89','CXX':'g++11.4.0','environment_script':'inputs/env.sh','environment_script_sha256':sha(D/'inputs/env.sh'),'test_root':PATHS['repo'],'all_binary_install_resolve_in_test_root':all(p.resolve().is_relative_to(Path(PATHS['repo']).resolve()) for p in (ROOT/'work/install').rglob('*') if p.is_file()),'final_gpu_resource_ref':'metadata/final-gpu-resource.txt','CPU_process_observation_in_evap_ref':'artifacts/evaporation-progress-resource-timeseries.json','flags_sources':'results tests[].argv, actualbuild logs/specs/versions; GPU objects originallynvcc -O2; fixturefresh -O3 --fmad=true sm89','no_profiling_or_nonprofiling_efficiency_remeasurement':'CurrentCUDA/Co codeunchanged; originalR2 Nsight/rawtimings reused withtheir old producer andscope.'}
(D/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
print('NATIVE_UNMODIFIED',git('diff','--name-only','--','applications','common','tests','tools'))
