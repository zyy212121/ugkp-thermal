from pathlib import Path
import resource,subprocess,json,time,hashlib
S=Path('/workspace/scratch/98b6339ec81b/pr14_fast4_evap_analysis');R=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');records=[]
for name,source,fixture in [('baseline_red',R/'applications/CHMT/film/CpuFilmDriver.C',S/'test_cpu_film_signed_open_edge.cpp'),('candidate_green',S/'CpuFilmDriver_signed_open_edge.C',S/'test_cpu_film_signed_open_edge.cpp'),('existing_original',R/'applications/CHMT/film/CpuFilmDriver.C',R/'applications/CHMT/devtools/multirate/test_cpu_film.cpp')]:
 binary=S/('test_cpu_film_'+name);cmd=['g++','-std=c++14','-O2','-D_GLIBCXX_ASSERTIONS','-Wall','-Wextra','-Werror','-pedantic','-I'+str(R/'applications/CHMT'),'-I'+str(R/'common'),str(fixture),str(source),'-o',str(binary)]
 for phase,argv in [('build',cmd),('run',[str(binary)])]:
  label=name+'_'+phase;start=time.monotonic()
  with (S/(label+'.stdout')).open('w') as out,(S/(label+'.stderr')).open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
  rec={'name':label,'argv':argv,'exit_code':p.returncode,'seconds':time.monotonic()-start,'maxrss_kib_cumulative_children':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss};records.append(rec);print(json.dumps(rec),flush=True)
  if phase=='build' and p.returncode:break
(S/'cpu_film_regression_receipt.json').write_text(json.dumps({'runs':records,'test_source_sha256':hashlib.sha256((S/'test_cpu_film_signed_open_edge.cpp').read_bytes()).hexdigest(),'tolerances':'Original near() default 2e-11 and all pre-existing thresholds unchanged. New direction/finite and timestep-equality assertions use exact comparisons.','scope':'Scratch-only component/native CPU film regression, not full native owner/GPU completion.'},indent=2)+'\n')
