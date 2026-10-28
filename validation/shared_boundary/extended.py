"""Native restart, physical budget, RK stage and unchanged-path timing checks."""
import importlib.util,json,sys,shutil,re,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('matrix',R/'validation/molecular_transport/native_matrix.py')
p=module('particles',R/'validation/transient_particles/check_native.py')
root=out.parent;report={'restart':{},'budgets':{},'timing':{}}
def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for solver,group,name in [('gasUGKP','gas','gasUGKP-table'),('gasUGKPSteady','gas','gasUGKPSteady-table'),('CHT','coupled','CHT-table'),('CHMT','coupled','CHMT-table')]:
 src=root/group/name
 report['restart'][solver]=m.resume(src,out/(solver+'-restart'),solver,R/'bin'/solver)
 print(solver,'restart PASS',flush=True);save()
for solver in ['gasUGKP','gasUGKPSteady']:
 src=root/'particles'/solver;dst=out/(solver+'-particle-restart')
 report['restart'][solver+'-particles']=m.resume(src,dst,solver,R/'bin'/solver)
 a=p.parcel(p.dirs(src)[-1]/'gpuResidentStrictParticles.dat');b=p.parcel(p.dirs(dst)[-1]/'gpuResidentStrictParticles.dat')
 assert len(a)==len(b)>0
 err=0
 for x,y in zip(a,b):
  assert x[-4:]==y[-4:]
  err=max(err,max(abs(u-v)/(1+abs(u)) for u,v in zip(x[:10],y[:10])))
 assert err<1e-8,err
 report['restart'][solver+'-particle-state']={'count':len(a),'max_scaled_error':err};save()
for name,mass,diam,steps,slope,velocity in [('hot_constant',1e-6,.001,1,0,0),('hot_nonlinear',1e-6,.001,1,-.22,0),('closed_drag_heat',.001,.001,100,0,2)]:
 c=out/name;p.one_cell(c,mass,diam,steps,slope,velocity);row=p.run(R/'bin/gasUGKP',c,True);assert row['exit']==0,row
 initial=p.budget(c/'0',mass,slope,True,velocity);final=p.budget(p.dirs(c)[-1],mass,slope);err=[abs(y-x) for x,y in zip(initial,final)]
 assert err[0]<1e-12 and err[1]<1e-10 and err[2]<1e-7,(name,err)
 report['budgets'][name]={'initial':initial,'final':final,'absolute_mass_momentum_energy_error':err};print(name,err,'PASS',flush=True);save()
for scheme in ['SSPRK3','Euler']:
 c=out/scheme;m.copy_input(root/'gas/gasUGKP-table',c)
 f=c/'system/fvSchemes';f.write_text(f.read_text().replace('default SSPRK2;','default '+scheme+';'))
 row=p.run(R/'bin/gasUGKP',c);assert row['exit']==0,row
 report[scheme]=row;save()
# Short-case elapsed wall times include initialization and output. No speedup claim.
for solver,source in [('gasUGKP',root/'smoke/legacy'),('CHT',root/'coupled/CHT-legacy'),('CHMT',root/'coupled/CHMT-legacy')]:
 if not (R/'build-evidence/shared-boundary/baseline-bin'/solver).exists():
  report['timing'][solver]={'skipped':'baseline binary unavailable'};save();continue
 values={'baseline':[],'candidate':[]}
 for repeat in range(3):
  for label,bindir in [('baseline',R/'build-evidence/shared-boundary/baseline-bin'),('candidate',R/'bin')]:
   c=out/(solver+'-'+label+'-'+str(repeat));m.copy_input(source,c)
   values[label].append(m.run([bindir/solver,'-case',c],c/'run.log'))
 values['median_ratio']=statistics.median(values['candidate'])/statistics.median(values['baseline'])
 report['timing'][solver]=values;print(solver,'timing',values,flush=True);save()
