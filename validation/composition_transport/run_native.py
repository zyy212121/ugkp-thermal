from pathlib import Path
import subprocess,json,time,os
import sys
R=Path(__file__).resolve().parents[2]
root=Path(sys.argv[1]).resolve();results=[]
for name in ['impermeable','uniform','nonuniform']:
 c=root/name
 for label,command in [('mesh',['blockMesh','-case',str(c)]),('steady',[str(R/'bin/gasUGKPSteady'),'-case',str(c)])]:
  start=time.monotonic()
  with (c/(label+'.log')).open('w') as out:p=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536'))
  results.append(dict(case=name,stage=label,returncode=p.returncode,seconds=time.monotonic()-start));(root/'runs.json').write_text(json.dumps(results,indent=2))
  print(results[-1],flush=True)
  if p.returncode:raise SystemExit(p.returncode)
