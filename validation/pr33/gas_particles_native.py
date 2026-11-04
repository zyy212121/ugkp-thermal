"""Current transient/steady mixture-particle owners, budgets and restart parity."""
import argparse,importlib.util,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native',ROOT/'validation/transient_particles/check_native.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);report={'runs':[],'budgets':[],'parity':[]}
 def save():(out/'results.json').write_text(json.dumps(report,indent=2))
 def run(case,solver,mesh=False):
  r=m.run(ROOT/'bin'/solver,case,mesh);r.update(case=case.name,solver=solver,sha256=m.sha(ROOT/'bin'/solver));report['runs'].append(r);save();assert r['exit']==0,r
 for name,mass,diam,steps,slope,velocity in [('hot_constant',1e-6,.001,1,0,0),('hot_nonlinear',1e-6,.001,1,-.22,0),('closed_drag_heat',.001,.001,100,0,2)]:
  d=out/name;m.one_cell(d,mass,diam,steps,slope,velocity);run(d,'gasUGKP',True)
  end=m.dirs(d)[-1];before=m.budget(d/'0',mass,slope,True,velocity);after=m.budget(end,mass,slope)
  error=[abs(x-y) for x,y in zip(before,after)];assert error[0]<1e-12 and error[1]<1e-10 and error[2]<1e-7,(name,error)
  tp=m.parcel(end/'gpuResidentStrictParticles.dat')[0][6];assert 100<tp<10000
  report['budgets'].append(dict(case=name,initial=before,final=after,absolute_mass_momentum_energy_error=error,particle_temperature=tp));save()
 for level in ['L0','L1','L2']:
  cases={}
  for solver in ['gasUGKP','gasUGKPSteady']:
   d=out/(level+'-'+solver);m.copy_inputs(ROOT/'validation/steady_cuda/evidence/cases/mixture_particles',d)
   m.control(d,200,steady=solver=='gasUGKPSteady');m.set_entry(d/'constant/schedulingProperties','gpuCsrLevel',level)
   run(d,solver);cases[solver]=d
  transient=m.dirs(cases['gasUGKP'])[-1];steady=m.dirs(cases['gasUGKPSteady'])[-1]
  parity=m.compare(transient,steady,32);parity['case']=level;report['parity'].append(parity);save()
  restart=out/(level+'-restart');m.copy_inputs(cases['gasUGKP'],restart)
  mid=next(p for p in m.dirs(cases['gasUGKP']) if abs(float(p.name)-.002)<1e-12);shutil.copytree(mid,restart/mid.name)
  m.set_entry(restart/'system/controlDict','startFrom','latestTime');run(restart,'gasUGKP')
  parity=m.compare(transient,m.dirs(restart)[-1],32);parity['case']=level+'-restart';report['parity'].append(parity);save()
 report['status']='PASS';save();print('PASS native gas/steady: three closed budgets, all CSR levels and restart')
if __name__=='__main__':main()
