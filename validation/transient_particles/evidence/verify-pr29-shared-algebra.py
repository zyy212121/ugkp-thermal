from pathlib import Path
import json,re,importlib.util
O=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/work/sutherland-wilke-20261026/transient-mixture-particles');r=O/'repo'
f=r/'validation/transient_particles/check_native.py';t=f.read_text().replace("'rhoY_A','rhoY_B']","'rhoY_A','rhoY_B','k','omega','nut']");f.write_text(t)
sp=importlib.util.spec_from_file_location('native',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
report=json.loads((O/'native-r3/results.json').read_text());checks=[]
for row in report['comparisons']:
 if 'reference_time' not in row:continue
 a=O/'native-r3'/(row['name']+'-steady_before')/row['reference_time'];b=O/'native-r3'/(row['name']+'-'+row['binary'])/row['actual_time']
 for name in ['k','omega','nut']:
  x=m.fields.internal(a/name,32);y=m.fields.internal(b/name,32);e=max(abs(i[0]-j[0])/(1+abs(i[0])) for i,j in zip(x,y));assert e<1e-8,(a,b,name,e)
  checks.append({'case':row['name'],'reference_time':row['reference_time'],'binary':row['binary'],'field':name,'max_scaled_error':e})
(O/'sst-output-check.json').write_text(json.dumps({'pass':True,'comparisons':checks},indent=2)+'\n')
# Verify the moved source is numerically unchanged after identifier/include adaptation.
import subprocess
old=subprocess.check_output(['git','show','ad8bc729:applications/gasUGKPSteady/private_backend/SteadyMixtureParticleCoupling.cuh'],cwd=r,text=True)
new=(r/'common/gasTransport/MixtureParticleCoupling.cuh').read_text()
old=old.replace('steadyMixtureParticles','mixtureParticles').replace('steadyMixtureParticleCouplingKernel','mixtureParticleCouplingKernel').replace('steadyParticleExternalKickKernel','mixtureParticleExternalKickKernel')
strip=lambda s:'\n'.join(l for l in s.splitlines() if not l.startswith('#include') and not l.startswith('//'))
assert strip(old)==strip(new)
print('80 saved-output pairs checked for k/omega/nut; shared exchange algebra unchanged.')
