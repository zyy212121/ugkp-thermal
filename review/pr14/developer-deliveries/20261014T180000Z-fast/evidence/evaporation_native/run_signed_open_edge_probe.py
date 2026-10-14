from pathlib import Path
import json,subprocess,resource,time,hashlib
S=Path('/workspace/scratch/98b6339ec81b/pr14_fast4_evap_analysis');R=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');checkpoint=R/'review/pr14/reviewer-deliveries/20261014T072053Z-fast/artifacts/extra/wet-evaporation-S209-D204/checkpoint-1'
receipts=[]
for name,source in [('original',R/'applications/CHMT/film/CpuFilmDriver.C'),('signed_open_edge',S/'CpuFilmDriver_signed_open_edge.C')]:
 exe=S/('boundary_probe_'+name);cmd=['g++','-std=c++17','-O0','-g0','-ffp-contract=off','-DUGKWP_GAS_SPECIES=2','-I'+str(R/'applications/CHMT'),'-I'+str(R/'common'),str(S/'checkpoint_film_boundary_probe.cpp'),str(source),'-o',str(exe)]
 for label,argv in [('build_'+name,cmd),('run_'+name,[str(exe),str(checkpoint),str(S/('boundary_'+name+'_state.json'))])]:
  start=time.monotonic()
  with (S/(label+'.stdout')).open('w') as out,(S/(label+'.stderr')).open('w') as err:result=subprocess.run(argv,stdout=out,stderr=err)
  r=resource.getrusage(resource.RUSAGE_CHILDREN);rec={'label':label,'argv':argv,'exit':result.returncode,'wall_seconds':time.monotonic()-start,'maxrss_kib_cumulative_children':r.ru_maxrss};receipts.append(rec);print(json.dumps(rec),flush=True)
  if result.returncode:break
report={'runs':receipts,'sources_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'checkpoint_film_boundary_probe.cpp',S/'CpuFilmDriver_signed_open_edge.C',R/'applications/CHMT/film/CpuFilmDriver.C',S/'signed_open_edge_scratch.patch']},'semantic_caveat':'Scratch-only deliberate extension to owner-extrapolated signed open-edge flux. Production test_cpu_film.cpp undeclaredInflowAndOutflow explicitly expects negative open-edge inflow to be rejected; no existing declared inflow BC is being invoked. No production patch applied.'}
(S/'signed_open_edge_execution_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
