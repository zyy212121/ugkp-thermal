"""Run built applications, checking capability rejection before solver advancement."""
from pathlib import Path
import importlib.util,sys,json,subprocess,os
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
s=importlib.util.spec_from_file_location('matrix',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
meta=json.loads((R/'validation/molecular_transport/fixtures.json').read_text())
report={}
for label,solver,fixture,reject in [('cht-single','CHT','cht-contact',False),('cht-mixture','CHT','cht-mixture',True),('chmt-frozen','CHMT','chmt-mixture',False),('chmt-single','CHMT','chmt-single',True)]:
 c=out/label;m.prepare(R/'validation/molecular_transport/fixtures'/fixture,c,meta[fixture])
 with (c/'run.log').open('w') as log:
  rc=subprocess.run([str(R/'bin'/solver),'-case',str(c)],stdout=log,stderr=subprocess.STDOUT,env=m.ENV).returncode
 text=(c/'run.log').read_text()
 assert (rc!=0)==reject,(label,rc,text[-3000:])
 if reject:assert ('CHT supports single-component' if solver=='CHT' else 'CHMT requires frozen multicomponent') in text
 report[label]={'exit_code':rc,'expected_rejection':reject};print(label,'PASS',flush=True)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
