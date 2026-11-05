import importlib.util,json,sys,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);root=Path(sys.argv[2]);report={'timing':{}}
s=importlib.util.spec_from_file_location('matrix',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def save():(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
# Short-case elapsed wall times include initialization and output. No speedup claim.
for solver,source in [('gasUGKP',root/'smoke/legacy'),('CHT',root/'coupled/CHT-legacy'),('CHMT',root/'coupled/CHMT-legacy')]:
 if not (R/'build-evidence/shared-boundary/baseline-bin'/solver).exists():
  report['timing'][solver]={'skipped':'baseline binary unavailable'};save();continue
 values={'baseline':[],'candidate':[]}
 for repeat in range(5):
  for label,bindir in [('baseline',R/'build-evidence/shared-boundary/baseline-bin'),('candidate',R/'bin')]:
   c=out/(solver+'-'+label+'-'+str(repeat));m.copy_input(source,c)
   values[label].append(m.run([bindir/solver,'-case',c],c/'run.log'))
 values['median_ratio']=statistics.median(values['candidate'])/statistics.median(values['baseline'])
 report['timing'][solver]=values;print(solver,'timing',values,flush=True);save()
