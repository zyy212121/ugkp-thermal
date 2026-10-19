import sys,json
C=json.loads(open(sys.argv[1]).read())
from pathlib import Path
import subprocess,json
R=Path(C['source']);E=Path(C['execution']);O=Path(C['logs']);P=Path(__file__).resolve().parent
t=(R/'validation/tests/gas_transport/gas_state_probe.cpp').read_text();src=E/'snapshot-budget.cpp';src.write_text(t[:t.index('void hashBytes')]+(P/'pr21-budget-driver.cpp').read_text())
cmd=['g++','-std=c++17','-O2','-I'+str(R),'-I'+str(R/'common'),'-I'+str(R/'common/gasNumerics'),'-I'+str(R/'validation/tests/gas_transport'),str(src),'-o',str(E/'snapshot-budget')]
r=subprocess.run(cmd,text=True,capture_output=True);(O/'snapshot-budget-build.log').write_text(r.stdout+r.stderr);print('BUILD',r.returncode,r.stderr[-3000:],flush=True);assert r.returncode==0
commands=[]
for case in ['baseline','wallfunction']:
 dest=O/(case+'-budget');dest.mkdir(exist_ok=True);args=[str(E/'snapshot-budget'),str(E/(case+'-018')),str(dest)];r=subprocess.run(args,text=True,capture_output=True);(dest/'run.log').write_text(r.stdout+r.stderr);commands.append({'command':args,'exit':r.returncode});print(case,'EXIT',r.returncode,r.stdout,r.stderr[-1200:],flush=True)
(O/'snapshot-budget-commands.json').write_text(json.dumps({'build':cmd,'build_exit':0,'runs':commands},indent=2))
