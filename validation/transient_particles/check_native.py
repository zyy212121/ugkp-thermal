#!/usr/bin/env python3
"""Native transient mixture/parcel acceptance and physical-time regression.
Source OpenFOAM's bashrc first; all output directories are created fresh.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,re,shutil,struct,subprocess,time
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
failure=module('heat_failure',ROOT/'validation/steady_cuda/check_heat_failure.py')
fields=module('fields',ROOT/'validation/steady_cpu/export_of.py')
ENV=dict(os.environ,OMP_NUM_THREADS='4',GAS_UGKP_CUDA_STACK_BYTES='65536')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def set_entry(path,name,value):
 t=path.read_text();pat=r'\b'+re.escape(name)+r'\s+[^;{}]+;'
 if re.search(pat,t):t=re.sub(pat,name+' '+str(value)+';',t)
 else:t+='\n'+name+' '+str(value)+';\n'
 path.write_text(t)
def control(case,steps,steady=False,write=20,dt=2e-5):
 p=case/'system/controlDict'
 for k,v in dict(application='gasUGKPSteady' if steady else 'gasUGKP',endTime=steps if steady else format(steps*dt,'.12g'),deltaT=1 if steady else dt,writeInterval=write,timePrecision=12).items():set_entry(p,k,v)
 set_entry(p,'windowSamples',20);set_entry(p,'samplingDeltaT',dt)
def run(binary,case,mesh=False):
 if mesh:
  with (case/'mesh.log').open('w') as log:p=subprocess.run(['blockMesh','-case',str(case)],env=ENV,stdout=log,stderr=subprocess.STDOUT)
  assert p.returncode==0,case
 cmd=[str(binary),'-case',str(case)];begin=time.monotonic()
 with (case/'run.log').open('w') as log:p=subprocess.run(cmd,env=ENV,stdout=log,stderr=subprocess.STDOUT)
 return {'command':cmd,'exit':p.returncode,'seconds':time.monotonic()-begin,'log_sha256':sha(case/'run.log')}
def dirs(case):
 return sorted((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?',p.name)),key=lambda p:float(p.name))
def parcel(path):
 with path.open('rb') as f:
  header=f.readline().decode().split();assert header[0]=='UGKP_PARTICLES_SCHEMA1_BIN'
  count=int(header[1]);rows=[]
  while len(rows)<count:
   n=struct.unpack('<I',f.read(4))[0];cols=[]
   for code in ['d']*10+['i']*2+['Q']*2:
    cols.append(struct.unpack('<'+str(n)+code,f.read(struct.calcsize('<'+str(n)+code))))
   rows.extend(zip(*cols))
 return sorted(rows,key=lambda x:x[-1])
FIELD_NAMES=['rho','rhoU','rhoE','U','p','T','epsilonS','rhoUs','rhoEs','rhoDs','Tp','theta','dMeanCell','Y_A','Y_B','rhoY_A','rhoY_B','k','omega','nut']
def compare(a,b,n):
 errors={}
 for directory in [a,b]:
  rho=fields.internal(directory/'rho',n);ya=fields.internal(directory/'rhoY_A',n);yb=fields.internal(directory/'rhoY_B',n)
  closure=max(abs(x[0]+y[0]-z[0]) for x,y,z in zip(ya,yb,rho))
  assert closure<1e-12,('species sum vs density',directory,closure)
 for name in FIELD_NAMES:
  x=fields.internal(a/name,n);y=fields.internal(b/name,n)
  maximum=max(abs(u-v)/(1+abs(u)) for xx,yy in zip(x,y) for u,v in zip(xx,yy));errors[name]=maximum
  assert maximum<1e-8,(name,maximum,a,b)
 pa=parcel(a/'gpuResidentStrictParticles.dat');pb=parcel(b/'gpuResidentStrictParticles.dat');assert len(pa)==len(pb)
 maxp=0
 for x,y in zip(pa,pb):
  assert x[-4:]==y[-4:],('parcel index/status/RNG/id changed',a,b)
  maxp=max(maxp,max(abs(u-v)/(1+abs(u)) for u,v in zip(x[:10],y[:10])))
 assert maxp<1e-8,('parcel fields',maxp)
 return {'fields_max_scaled_error':errors,'parcels':len(pa),'parcel_max_scaled_error':maxp}
def copy_inputs(source,d):
 d.mkdir()
 for name in ['0','constant','system']:shutil.copytree(source/name,d/name)
def one_cell(d,mass,diameter,steps=1,cp_slope=0,velocity=0):
 failure.prepare(d,mass,diameter,1,100);control(d,steps,write=1)
 if cp_slope:
  f=d/'constant/gasModelProperties';f.write_text(f.read_text().replace('(1000 0 0)',f'(1000 {cp_slope} 0)'))
 if velocity:
  p=d/'0/gpuResidentStrictParticles.dat';b=bytearray(p.read_bytes());start=b.index(b'\n')+1+4;struct.pack_into('<d',b,start+3*8,velocity);p.write_bytes(b)
 # Suppress boundary heat/momentum flux: periodic uniform one-cell state.
 f=d/'system/blockMeshDict';t=f.read_text().replace('left { type patch;','left { type cyclic; neighbourPatch right;').replace('right { type patch;','right { type cyclic; neighbourPatch left;');f.write_text(t)
 for f in (d/'0').iterdir():
  if f.suffix=='.dat':continue
  t=f.read_text().replace('left { type zeroGradient; }','left { type cyclic; }').replace('right { type zeroGradient; }','right { type cyclic; }');f.write_text(t)
def budget(directory,mass,slope=0,initial=False,velocity=0):
 V=.7;eps=mass/(2500*V)
 if initial:return [V*(1-eps)+mass,V*(1-eps)*0+mass*velocity,V*(1-eps)*(1000*1000+.5*slope*1e6-8.31446261815324/.028*1000)+mass*(800*10000+.5*velocity**2)]
 scalar=lambda name:fields.internal(directory/name,1)[0][0]
 gasrho=scalar('rho');eps=scalar('epsilonS');rhoU=fields.internal(directory/'rhoU',1)[0]
 ps=parcel(directory/'gpuResidentStrictParticles.dat')
 return [V*(1-eps)*gasrho+sum(x[9] for x in ps),V*(1-eps)*rhoU[0]+sum(x[9]*x[3] for x in ps),V*(1-eps)*scalar('rhoE')+sum(x[9]*(800*x[6]+.5*sum(v*v for v in x[3:6])+1.5*x[7]) for x in ps)]
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ['candidate','before','steady_before','steady_after','output']:ap.add_argument('--'+name.replace('_','-'),type=Path,required=True)
 a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 binaries={k:getattr(a,k).resolve() for k in ['candidate','before','steady_before','steady_after']}
 report={'binaries':{k:{'path':str(v),'sha256':sha(v)} for k,v in binaries.items()},'runs':[],'comparisons':[]}
 def save(): (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 def execute(label,key,d,mesh=False,expected=0):
  row=run(binaries[key],d,mesh);row.update(label=label,binary=key,expected_exit=expected);report['runs'].append(row);save();print(label,row['exit'],round(row['seconds'],3),flush=True);assert (row['exit']==0)==(expected==0),(label,row);return row
 # Old transient frontend rejects the same valid mixture/particle input.
 d=out/'before-rejected';one_cell(d,1e-6,.001);execute(d.name,'before',d,True,1);assert 'Unsupported shared gas model' in (d/'run.log').read_text()
 # Final physical step reports the real heat failure, not a next-step error.
 d=out/'heat-failure';one_cell(d,1.,1e-6);execute(d.name,'candidate',d,True,1);assert len(dirs(d))==1 and 'detailCode=4' in (d/'run.log').read_text() and 'site=30700' in (d/'run.log').read_text()
 for name,mass,diam,steps,slope,velocity in [('hot_constant',1e-6,.001,1,0,0),('hot_nonlinear',1e-6,.001,1,-.22,0),('closed_drag_heat',.001,.001,100,0,2)]:
  d=out/name;one_cell(d,mass,diam,steps,slope,velocity);execute(name,'candidate',d,True)
  final=dirs(d)[-1];b0=budget(d/'0',mass,slope,True,velocity);b1=budget(final,mass,slope);err=[abs(y-x) for x,y in zip(b0,b1)]
  assert err[0]<1e-12 and err[1]<1e-10 and err[2]<1e-7,(name,err)
  ps=parcel(final/'gpuResidentStrictParticles.dat');assert ps and ps[0][6]<10000 and ps[0][6]>100
  ya=fields.internal(final/'rhoY_A',1)[0][0];yb=fields.internal(final/'rhoY_B',1)[0][0];rho=fields.internal(final/'rho',1)[0][0];assert abs(ya+yb-rho)<1e-12
  report['comparisons'].append({'name':name,'initial_budget':b0,'final_budget':b1,'absolute_mass_momentum_energy_error':err,'parcel_temperature':ps[0][6]});save()
 # All three generic parcel directory modes, plus heterogeneous transport/SST.
 specs=[('mixture_L0','mixture_particles','L0',False),('mixture_L1','mixture_particles','L1',False),('mixture_L2_wilke','mixture_particles','L2',True),('mixture_sst','mixture_particles','L1',False)]
 for name,fixture,level,wilke in specs:
  prototype=ROOT/'validation/steady_cuda/evidence/cases'/fixture
  cases={}
  for key in ['steady_before','candidate','steady_after']:
   d=out/(name+'-'+key);copy_inputs(prototype,d);control(d,200,steady=key!='candidate');set_entry(d/'constant/schedulingProperties','gpuCsrLevel',level)
   if wilke:(d/'constant/gasTransportProperties').write_text('FoamFile { version 2.0; format ascii; class dictionary; object gasTransportProperties; }\nviscosityModel sutherland; species (A B); sutherland { A { muRef 1.8e-5; TRef 300; S 110; molarMass .028; } B { muRef 2.1e-5; TRef 300; S 140; molarMass .028; } }\n')
   if wilke:
    set_entry(d/'constant/particleProperties','parcelMass',1e-8);set_entry(d/'constant/particleProperties','injectionParcelMass',1e-8)
   if name=='mixture_sst':shutil.copy2(ROOT/'validation/steady_cuda/evidence/cases/mixture_sst/constant/fluidProperties',d/'constant/fluidProperties')
   execute(d.name,key,d);cases[key]=d
  for key in ['candidate','steady_after']:
   ref=dirs(cases['steady_before'])[1:];actual=dirs(cases[key])[1:];assert len(ref)==len(actual)
   for aa,bb in zip(ref,actual):
    if key=='candidate':assert abs(float(bb.name)-float(aa.name)*2e-5)<1e-12
    row=compare(aa,bb,32);row.update(name=name,reference_time=aa.name,actual_time=bb.name,binary=key);report['comparisons'].append(row)
  # Species must evolve and be saved, not remain equal to initialization.
  final=dirs(cases['candidate'])[-1];initial=fields.internal(cases['candidate']/'0/Y_A',32);actual=fields.internal(final/'Y_A',32)
  assert max(abs(x[0]-y[0]) for x,y in zip(initial,actual))>1e-6
  # Restart halfway through physical time, preserving parcel RNG and weights.
  restart=out/(name+'-restart');copy_inputs(cases['candidate'],restart)
  mid=next(p for p in dirs(cases['candidate']) if abs(float(p.name)-.002)<1e-12);shutil.copytree(mid,restart/mid.name)
  set_entry(restart/'system/controlDict','startFrom','latestTime');execute(restart.name,'candidate',restart)
  row=compare(final,dirs(restart)[-1],32);row.update(name=name+'-restart');report['comparisons'].append(row);save()
 report['pass']=True;save()
if __name__=='__main__':main()