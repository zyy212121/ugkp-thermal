import sys,json
C=json.loads(open(sys.argv[1]).read())
from pathlib import Path
import sys,importlib.util,re,json,csv
import numpy as np
O=Path(C['logs']);E=Path(C['execution']);E.mkdir(exist_ok=True)
import foam_io as m
cases={'baseline':Path(C['snapshots'])/'baseline-boundaryLayer','wallfunction':Path(C['snapshots'])/'wallFunction-f9'};B=cases['baseline'];pm=B/'constant/polyMesh'
points=np.array([list(map(float,x.split()))for x in re.findall(r'\(([^()]*)\)',m.list_body((pm/'points').read_text()))]);faces=np.array([list(map(int,x.split()))for x in re.findall(r'\d+\(([^()]*)\)',m.list_body((pm/'faces').read_text()))]);assert faces.shape[1]==4
P=points[faces];Cf=P.mean(axis=1);Sf=np.cross(P,np.roll(P,-1,axis=1)).sum(axis=1)/2;area=np.linalg.norm(Sf,axis=1);normal=Sf/area[:,None]
own=np.array(list(map(int,m.list_body((pm/'owner').read_text()).split())),dtype=np.int32);nei0=np.array(list(map(int,m.list_body((pm/'neighbour').read_text()).split())),dtype=np.int32);nf=len(own);ni=len(nei0);nei=np.r_[nei0,np.full(nf-ni,-1,dtype=np.int32)]
C=m.read_field(B/'0/C',vector=True);V=m.read_field(B/'0/V');nc=len(C)
delta=Cf-C[own];delta[:ni]=C[nei[:ni]]-C[own[:ni]];distance=np.sum(normal*delta,axis=1);assert distance.min()>0
coeff=1/distance;weight=np.ones(nf);weight[:ni]=np.sum(normal[:ni]*(C[nei[:ni]]-Cf[:ni]),axis=1)/distance[:ni]
cellids=np.r_[own,nei0];fids=np.r_[np.arange(nf),np.arange(ni)];order=np.argsort(cellids,kind='stable');counts=np.bincount(cellids,minlength=nc).astype(np.int32);starts=np.r_[0,np.cumsum(counts[:-1])].astype(np.int32);cellfaces=fids[order].astype(np.int32)
volumecheck=np.bincount(own,weights=np.sum(Sf*(Cf-C[own]),axis=1)/3,minlength=nc)+np.bincount(nei0,weights=-np.sum(Sf[:ni]*(Cf[:ni]-C[nei0]),axis=1)/3,minlength=nc);relative=np.max(abs(volumecheck/V-1));assert relative<1e-10
geometry={'faceOwner':own,'faceNeighbour':nei,'cellPlaneCount':counts,'cellPlaneStart':starts,'cellFaceId':cellfaces,'facePeriodicPair':np.full(nf,-1,dtype=np.int32),'V':V,'cellLength':V**(1/3),'deltaCoeffs':coeff,'faceWeight':weight,'magSf':area,'sstWallDistance':C[:,1]}
for j,d in enumerate('xyz'):
 geometry['C'+d]=C[:,j];geometry['faceC'+d]=Cf[:,j];geometry['Sf'+d]=Sf[:,j]
patchtext=m.strip_comments((pm/'boundary').read_text());patches={}
for match in re.finditer(r'(\w+)\s*\{([^{}]*)\}',patchtext):
 name,body=match.groups();a=re.search(r'startFace\s+(\d+)',body);b=re.search(r'nFaces\s+(\d+)',body)
 if a and b:patches[name]=(int(a[1]),int(b[1]))
for label,case in cases.items():
 out=E/(label+'-018');out.mkdir(exist_ok=True);data={k:v.copy()for k,v in geometry.items()};tp=m.numeric_dirs(case)[.018]
 for name,target in [('rho','rho'),('T','Tgas'),('p','p'),('k','k'),('omega','omega'),('nut','nut'),('rhoE','rhoE')]:data[target]=m.read_field(tp/name,nc)
 for name,target in [('U','U'),('rhoU','rhoU')]:
  ar=m.read_field(tp/name,nc,True)
  for j,d in enumerate('xyz'):data[target+d]=ar[:,j]
 for sp in ['Air','CO2']:data['speciesRho'+sp]=m.read_field(tp/('rhoY_'+sp),nc)
 kinds=np.zeros(nf,dtype=np.int32)
 for name in ['upstreamWall','porousWall','downstreamWall']:
  start,num=patches[name];kinds[start:start+num]=2
 start,num=patches['sides'];kinds[start:start+num]=3
 for field in ['gasBoundaryKind','riemannBoundaryKind']:data[field]=kinds
 for prefix in ['riemannBoundary','gasBoundary']:
  for suffix in ['PFix','TFix','UFix','RhoFix']:data[prefix+suffix]=np.zeros(nf,dtype=np.int32)
  for suffix in ['P','T','Ux','Uy','Uz','Rho']:data[prefix+suffix]=np.zeros(nf)
 for name,(start,num)in patches.items():
  if name=='sides':continue
  for prefix in ['riemannBoundary','gasBoundary']:
   if 'Wall'in name:
    data[prefix+'T'][start:start+num]=290.53216418474562;data[prefix+'TFix'][start:start+num]=1;data[prefix+'UFix'][start:start+num]=1
   else:
    data[prefix+'P'][start:start+num]=124094.11053558919;data[prefix+'PFix'][start:start+num]=1
    if name!='outlet':
     data[prefix+'T'][start:start+num]=275.90898782976797;data[prefix+'TFix'][start:start+num]=1;data[prefix+'UFix'][start:start+num]=1;data[prefix+'Ux'][start:start+num]=179.81233357102013
 data['sstBoundaryKMode']=np.zeros(nf,dtype=np.int32);data['sstBoundaryOmegaMode']=np.zeros(nf,dtype=np.int32);data['sstBoundaryK']=np.zeros(nf);data['sstBoundaryOmega']=np.zeros(nf)
 for name in ['inlet','top']:
  start,num=patches[name];data['sstBoundaryKMode'][start:start+num]=1;data['sstBoundaryOmegaMode'][start:start+num]=1;data['sstBoundaryK'][start:start+num]=1.212467823909593;data['sstBoundaryOmega'][start:start+num]=2010.3630052883425
 for name,array in data.items():array.astype('<i4'if np.issubdtype(array.dtype,np.integer)else'<f8').tofile(out/(name+'.bin'))
 (out/'counts.txt').write_text(f'{nc} {nf} {ni}\n');(out/'geometry-validation.json').write_text(json.dumps({'volume_max_relative_difference':float(relative),'cells':nc,'faces':nf,'internal_faces':ni,'case':str(case),'time':str(tp),'source':'Read-only snapshot reconstruction; geometric planes are orthogonal quads; not an original GPU RK-stage capture.'},indent=2))
 if label=='baseline':
  flux=list(csv.DictReader((tp/'uniform/gasBoundaryLayer.csv').open()));np.array([[int(r['face']),int(r['owner']),float(r['traceDensity'])*287.0500676*290.53216418474562,float(r['heatFluxIntoGas']),float(r['tractionX']),float(r['tractionY']),float(r['tractionZ']),float(r['ownerOmega']),float(r['ownerDistance'])]for r in flux],dtype='<f8').tofile(out/'wall-exchange.bin')
 print('EXPORTED',out,nc,nf,flush=True)
(O/'budget-export.json').write_text(json.dumps({'path':str(E),'geometry_max_relative_volume_difference':float(relative)},indent=2))
