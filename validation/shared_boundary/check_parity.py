import sys,importlib.util,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('m',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report={}
for solver,source in [('gasUGKP','smoke/legacy'),('CHT','coupled/CHT-legacy'),('CHMT','coupled/CHMT-legacy')]:
 candidate=Path(sys.argv[2])/source;c=out/solver;m.copy_input(candidate,c)
 seconds=m.run([R/'build-evidence/shared-boundary/baseline-bin'/solver,'-case',c],c/'run.log')
 errors=m.compare_chmt(c,candidate) if solver=='CHMT' else m.compare_fields(m.times(c)[-1],m.times(candidate)[-1])
 report[solver]={'errors':errors,'baselineSeconds':seconds};print(solver,'PASS',flush=True)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
