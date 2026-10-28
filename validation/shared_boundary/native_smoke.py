import os,shutil,re,subprocess,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
src=Path(__file__).resolve().parent/'fixtures/gas-legacy'
report={}
for name in ['legacy','total']:
 c=out/name;c.mkdir()
 for part in ['0','constant','system']:shutil.copytree(src/part,c/part)
 p=c/'system/controlDict';s=p.read_text();s=s.replace('application gasUGKPSteady','application gasUGKP').replace('endTime 2;','endTime 2e-7;').replace('deltaT 1;','deltaT 1e-7;');p.write_text(s)
 if name=='total':
  for field,block in [('U','type pressureInletOutletVelocity; value uniform (70 0 0);'),('p','type totalPressure; p0 uniform 104200; psi thermo:psi; gamma 1.4; value uniform 101325;'),('T','type totalTemperature; T0 uniform 302.5; gamma 1.4; value uniform 300;')]:
   p=c/'0'/field;s,n=re.subn(r'inlet\s*\{[^}]*\}','inlet { '+block+' }',p.read_text());assert n==1;p.write_text(s)
 with (c/'run.log').open('w') as log:
  status=subprocess.run([R/'bin/gasUGKP','-case',c],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536')).returncode
 report[name]=status;print(name,status,flush=True)
 if status:print((c/'run.log').read_text()[-3500:]);break
(out/'results.json').write_text(json.dumps(report,indent=2))
assert report=={'legacy':0,'total':0},report
