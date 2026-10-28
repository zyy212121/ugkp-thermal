from pathlib import Path
import sys,os,subprocess,time,shutil,json,statistics,hashlib
O=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/steady-pr-review')
W=O.parents[1]/'steady-validation/repair'
sys.path.insert(0,str(W/'source/validation/steady_cpu'))
from export_of import export
from compare_application import comparison
F=O/'failure-timing';F.mkdir(exist_ok=True)
env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536',OMP_NUM_THREADS='4')
report={'scope':'Paired 3000-iteration native mixture+particle case, same fast compile flags; includes process startup and output; no production-optimization claim','runs':[]}
executables={'before':O/'fixed-bin/gasUGKPSteady','after':O/'failure-fixed-r3-bin/gasUGKPSteady'}
report['binaries']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in executables.items()}
for repetition in range(3):
 for label in (['before','after'] if repetition%2==0 else ['after','before']):
  case=F/f'{label}-{repetition}';shutil.copytree(O/'source/validation/steady_cuda/evidence/cases/mixture_particles',case)
  shutil.copy2(W/'cases/mixture_particles/reference-input.json',case/'reference-input.json')
  control=case/'system/controlDict';control.write_text(control.read_text().replace('endTime 20000','endTime 3000'))
  start=time.monotonic()
  with (case/'run.log').open('w') as log:p=subprocess.run([str(executables[label]),'-case',str(case)],env=env,stdout=log,stderr=subprocess.STDOUT)
  row={'label':label,'repetition':repetition,'seconds':time.monotonic()-start,'exit_code':p.returncode};report['runs'].append(row)
  assert p.returncode==0,row
  export(case,F/f'{label}-{repetition}.csv')
  (F/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  print(row,flush=True)
 fields=['rho','rhoUx','rhoUy','rhoUz','rhoE','Ux','Uy','Uz','p','T','rhoY0','rhoY1','rhoSolid','rhoUsx','rhoUsy','rhoUsz','rhoEs','rhoHp','rhoDs']
 comparison_result=comparison(F/f'before-{repetition}.csv',F/f'after-{repetition}.csv',fields,1e-8,1e-10,start=3000,end=3000)
 report.setdefault('comparisons',[]).append(comparison_result);assert comparison_result['pass']
report['median_seconds']={k:statistics.median(r['seconds'] for r in report['runs'] if r['label']==k) for k in executables}
report['speedup']=report['median_seconds']['before']/report['median_seconds']['after'];report['status']='PASS'
(F/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['median_seconds'],report['speedup'],flush=True)
