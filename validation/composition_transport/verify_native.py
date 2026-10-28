#!/usr/bin/env python3
import csv,json,math,sys,re,shutil,subprocess,os,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[2];root=Path(sys.argv[1]).resolve()
spec=importlib.util.spec_from_file_location('matrix',R/'validation/molecular_transport/native_matrix.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report={'wall':{}}
for name in ['impermeable','uniform','nonuniform']:
 c=root/name;expected=json.loads((c/'expected-injection.json').read_text());maximum={'massFluxError':0.,'speciesFluxError':0.,'heatBalanceAbsolute':0.,'viscosityRelativeError':0.,'wallResidual':0.};traces=[]
 assert 'v1_0;' not in (c/'2/gasTransportIdentity').read_text()
 for t in ['1','2']:
  diag=list(csv.DictReader((c/t/'uniform/gasBoundaryLayer.csv').open()))
  profiles=list(csv.DictReader((c/t/'uniform/gasBoundaryLayerProfiles.csv').open()));wall={int(x['face']):x for x in profiles if int(x['node'])==0}
  assert len(diag)==len(expected)
  for j,d in zip(expected,diag):
   w={k:float(v) for k,v in wall[int(d['face'])].items()};d={k:float(v) for k,v in d.items()}
   mass=d['traceDensity']*sum(w['U'+a.lower()]*w['normal'+a] for a in 'xyz'.upper())
   maximum['massFluxError']=max(maximum['massFluxError'],abs(mass-j))
   maximum['speciesFluxError']=max(maximum['speciesFluxError'],abs(d['speciesFlux_Air']),abs(d['speciesFlux_CO2']-j))
   heat=d['heatFluxIntoGas']-d['heatConduction']-d['heatViscous']-d['heatTransport'];maximum['heatBalanceAbsolute']=max(maximum['heatBalanceAbsolute'],abs(heat))
   T,y=w['T'],w['Y_CO2'];mu=(17.10-4.65*y+1.20*y*y)*1e-6*(T/273)**1.5*(273+114+220*y)/(T+114+220*y)
   maximum['viscosityRelativeError']=max(maximum['viscosityRelativeError'],abs(d['viscosity']/mu-1));maximum['wallResidual']=max(maximum['wallResidual'],d['residual'])
   assert abs(w['Y_Air']+y-1)<1e-14 and 0<=y<1
   traces.append(y)
 assert maximum['massFluxError']<1e-13 and maximum['speciesFluxError']<1e-14
 assert maximum['heatBalanceAbsolute']<1e-7 and maximum['viscosityRelativeError']<1e-13
 report['wall'][name]=dict(maximum,traceCO2Range=[min(traces),max(traces)])
# Additional independent configurations use fresh native processes.
def clone(name,source='nonuniform'):
 c=root/name;c.mkdir();src=root/source
 for part in ['0','constant','system']:shutil.copytree(src/part,c/part)
 return c
def run(c,label='run',reject=False):
 with (c/(label+'.log')).open('w') as f:p=subprocess.run([str(R/'bin/gasUGKPSteady'),'-case',str(c)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536'))
 if reject:
  assert p.returncode!=0 and 'identity' in (c/(label+'.log')).read_text();return
 assert p.returncode==0,(c,p.returncode)
restart=clone('restart');shutil.copytree(root/'nonuniform/1',restart/'1');m.entry(restart/'system/controlDict','startTime',1);run(restart)
report['restartRelativeErrors']=m.compare_fields(root/'nonuniform/2',restart/'2',restart=True)
changed=clone('changed-D');shutil.copytree(root/'nonuniform/1',changed/'1');m.entry(changed/'system/controlDict','startTime',1)
p=changed/'constant/gasTransportProperties';p.write_text(p.read_text().replace('temperatureExponent 1.81','temperatureExponent 1.82'));run(changed,reject=True);report['changedDCoefficientRejectedOnRestart']=True
# No constant diffusion allocation is needed when a power law is configured.
none=clone('diffusion-none');p=none/'constant/gasModelProperties';p.write_text(p.read_text().replace('model constant; coefficients (0 0);','model none;'));run(none)
report['diffusionNoneParity']=m.compare_fields(root/'nonuniform/2',none/'2')
# Both representations of a constant wall field have identical production results.
uniform=clone('uniform-fields','uniform');p=uniform/'constant/gasWallProperties';p.write_text(p.read_text().replace('speciesFlux (0 .1);','speciesFluxFields { Air uniform 0; CO2 uniform .1; }'));run(uniform)
report['uniformEncodingParity']=m.compare_fields(root/'uniform/2',uniform/'2')
(root/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
