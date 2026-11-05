#!/usr/bin/env python3
"""Closed periodic transient FV mass/species/momentum/energy budget with variable transport."""
from pathlib import Path
import sys,importlib.util,shutil,re,math,json,subprocess,os
R=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('fields',R/'validation/steady_cpu/export_of.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
source=R/'validation/steady_cuda/evidence/cases/mixture_diffusion'
for part in ['0','constant','system']:shutil.copytree(source/part,out/part)
p=out/'system/blockMeshDict';p.write_text(p.read_text().replace('left { type patch;','left { type cyclic; neighbourPatch right;').replace('right { type patch;','right { type cyclic; neighbourPatch left;'))
for p in (out/'0').iterdir():
 if p.is_file():p.write_text(re.sub(r'\b(left|right)\s*\{[^{}]*\}',lambda m:m[1]+' { type cyclic; }',p.read_text()))
text=(Path(__file__).with_name('transport.dict')).read_text();text=re.sub(r'\b(Air|CO2)\b',lambda m:'A' if m[1]=='Air' else 'B',text);(out/'constant/gasTransportProperties').write_text(text)
p=out/'system/controlDict';text=p.read_text()
for k,v in dict(application='gasUGKP',endTime='0.0004',deltaT='0.00002',writeInterval='20',writeControl='timeStep',adjustTimeStep='false',timePrecision='12').items():text=re.sub(r'\b'+k+r'\s+[^;]+;',k+' '+v+';',text)
p.write_text(text)
for label,cmd in [('mesh',['blockMesh','-case',str(out)]),('run',[str(R/'bin/gasUGKP'),'-case',str(out)])]:
 with (out/(label+'.log')).open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,GAS_UGKP_CUDA_STACK_BYTES='65536'))
 assert p.returncode==0,(label,p.returncode)
def scalar(d,name):return [row[0] for row in f.internal(d/name,32)]
def budget(d,initial=False):
 rho=scalar(d,'rho');Y=[scalar(d,'Y_A'),scalar(d,'Y_B')];U=f.internal(d/'U',32);energy=scalar(d,'rhoE')
 momentum=[[rho[i]*U[i][j] for i in range(32)] for j in range(3)] if initial else list(zip(*f.internal(d/'rhoU',32)))
 species=[[rho[i]*y[i] for i in range(32)] for y in Y] if initial else [scalar(d,'rhoY_A'),scalar(d,'rhoY_B')]
 return [math.fsum(a)*2/32 for a in [rho,*species,*momentum,energy]]
a=budget(out/'0',True);times=sorted((p for p in out.iterdir() if p.is_dir() and re.fullmatch(r'[\d.eE+-]+',p.name) and float(p.name)>0),key=lambda p:float(p.name));b=budget(times[-1]);errors=[abs(x-y) for x,y in zip(a,b)]
assert all(e<1e-10*max(abs(x),1) for e,x in zip(errors,a)),(a,b,errors)
change=max(abs(x-y) for x,y in zip(scalar(out/'0','Y_A'),scalar(times[-1],'Y_A')));assert change>1e-12
report=dict(time=times[-1].name,steps=20,quantities=['mass','species_A','species_B','momentum_x','momentum_y','momentum_z','total_energy'],initial=a,final=b,absoluteError=errors,maximumSpeciesChange=change)
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
