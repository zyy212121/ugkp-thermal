import sys,json
C=json.loads(open(sys.argv[1]).read())
from pathlib import Path
import numpy as np
import subprocess,json,csv
R=Path(C['source']);E=Path(C['execution']);O=Path(C['logs']);M=E/'manufactured-shear';M.mkdir(exist_ok=True)
nu=1.7244421537657099e-5;kappa=.41;wallE=9.8
def uplus(yplus):
 yplus=np.asarray(yplus);lo=np.zeros_like(yplus);hi=np.minimum(yplus,np.log(1+9.8*yplus)/.41+5)
 for _ in range(80):
  mid=(lo+hi)/2;z=.41*mid;yp=mid+(np.exp(z)-1-z-z*z/2-z*z*z/6)/9.8;mask=yp>yplus;hi=np.where(mask,mid,hi);lo=np.where(mask,lo,mid)
 return (lo+hi)/2
def eddy(y):
 z=kappa*uplus(y/nu);return nu*kappa/wallE*(np.exp(z)-1-z-z*z/2)
gauss,weights=np.polynomial.legendre.leggauss(64);allrows=[];commands=[]
for target in [30,80,200]:
 H=2*target*nu*16;fixedY=2*target*nu
 for N in [16,32,64,128,256,512,1024]:
  root=M/f'yp{target}-n{N}';root.mkdir(exist_ok=True);out=O/'manufactured-shear'/f'yp{target}-n{N}';out.mkdir(parents=True,exist_ok=True);h=H/N
  y=(np.arange(N)+.5)*h;U=(uplus(((np.arange(N)[:,None]+.5)*h+gauss[None,:]*h/2)/nu)*weights[None,:]).sum(axis=1)/2
  Cf=[];Sf=[];owner=[];neighbor=[];kind=[]
  def face(c,n,cf,sf,k):Cf.append(cf);Sf.append(sf);owner.append(c);neighbor.append(n);kind.append(k)
  for i in range(N-1):face(i,i+1,[.5,(i+1)*h,.5],[0,1,0],0)
  ni=N-1;face(0,-1,[.5,0,.5],[0,-1,0],2);face(N-1,-1,[.5,H,.5],[0,1,0],0)
  for i in range(N):
   face(i,-1,[0,y[i],.5],[-h,0,0],0);face(i,-1,[1,y[i],.5],[h,0,0],0);face(i,-1,[.5,y[i],0],[0,0,-h],3);face(i,-1,[.5,y[i],1],[0,0,h],3)
  Cf=np.array(Cf);Sf=np.array(Sf);owner=np.array(owner,dtype=np.int32);neighbor=np.array(neighbor,dtype=np.int32);kind=np.array(kind,dtype=np.int32);nf=len(owner);C=np.column_stack([np.full(N,.5),y,np.full(N,.5)]);area=np.linalg.norm(Sf,axis=1);normal=Sf/area[:,None];delta=Cf-C[owner];delta[:ni]=C[neighbor[:ni]]-C[owner[:ni]];dist=(normal*delta).sum(axis=1)
  data={'faceOwner':owner,'faceNeighbour':neighbor,'facePeriodicPair':np.full(nf,-1,dtype=np.int32),'faceWeight':np.ones(nf),'magSf':area,'deltaCoeffs':1/dist,'V':np.full(N,h),'cellLength':np.full(N,h**(1/3)),'sstWallDistance':y,'rho':np.ones(N),'p':np.full(N,287.0500676*300),'Tgas':np.full(N,300.),'Ux':U,'Uy':np.zeros(N),'Uz':np.zeros(N),'rhoUx':U,'rhoUy':np.zeros(N),'rhoUz':np.zeros(N),'rhoE':np.full(N,(1004.675-287.0500676)*300)+.5*U*U,'k':np.full(N,1/np.sqrt(.09)),'omega':1/(np.sqrt(.09)*kappa*y),'nut':eddy(y),'speciesRhoAir':np.ones(N),'speciesRhoCO2':np.zeros(N)}
  data['faceWeight'][:ni]=.5
  for d in range(3):data['C'+'xyz'[d]]=C[:,d];data['faceC'+'xyz'[d]]=Cf[:,d];data['Sf'+'xyz'[d]]=Sf[:,d]
  cl=[[]for _ in range(N)]
  for f,c in enumerate(owner):cl[c].append(f)
  for f,c in enumerate(neighbor[:ni]):cl[c].append(f)
  count=np.array([len(x)for x in cl],dtype=np.int32);data.update(cellPlaneCount=count,cellPlaneStart=np.r_[0,np.cumsum(count[:-1])].astype(np.int32),cellFaceId=np.array([f for row in cl for f in row],dtype=np.int32),gasBoundaryKind=kind,riemannBoundaryKind=kind)
  for prefix in ['riemannBoundary','gasBoundary']:
   data[prefix+'UFix']=np.array([1 if f>=ni and kind[f]!=3 else 0 for f in range(nf)],dtype=np.int32);data[prefix+'TFix']=data[prefix+'UFix'].copy();data[prefix+'PFix']=np.array([1 if f>=ni and kind[f]==0 else 0 for f in range(nf)],dtype=np.int32);data[prefix+'RhoFix']=np.zeros(nf,dtype=np.int32);data[prefix+'T']=np.full(nf,300.);data[prefix+'P']=np.full(nf,287.0500676*300);data[prefix+'Ux']=U[owner].copy();data[prefix+'Ux'][ni]=0;data[prefix+'Ux'][ni+1]=float(uplus(H/nu));data[prefix+'Uy']=np.zeros(nf);data[prefix+'Uz']=np.zeros(nf);data[prefix+'Rho']=np.ones(nf)
  for name,value in data.items():value.astype('<i4'if np.issubdtype(value.dtype,np.integer)else'<f8').tofile(root/(name+'.bin'))
  (root/'counts.txt').write_text(f'{N} {nf} {ni}\n');definition={'purpose':'Manufactured 1-D momentum transport only, NOT an exact full coupled SST/energy solution. Prescribed Spalding U(y) and nu_eff(y)=nu*dyPlus/duPlus give exact rho*nu_eff*dU/dy=1 Pa; cell velocities are quadrature cell averages.','target_first_cell_yplus_at_N16':target,'utau':1,'rho':1,'nu':nu,'domain_height':H,'cells':N,'fixed_physical_comparison_face_y':fixedY,'expected_physical_shear':1};(root/'definition.json').write_text(json.dumps(definition,indent=2))
  args=[str(E/'snapshot-budget'),str(root),str(out),'1'];r=subprocess.run(args,text=True,capture_output=True);(out/'run.log').write_text(r.stdout+r.stderr);commands.append({'command':args,'exit':r.returncode});assert r.returncode==0,r.stderr
  rows=list(csv.DictReader((out/'normal-face-budget.csv').open()));v=min(rows,key=lambda v:abs(float(v['y'])-fixedY));assert abs(float(v['y'])-fixedY)<1e-10
  entry={**definition,**{key:float(v[key])for key in ['y','full_mx','inviscid_mx','viscous_mx','advective_mx','numerical_mx','HLLC_constant_gamma_core_mx','SLAU2_constant_gamma_core_mx']}};allrows.append(entry);print('PROBE',target,N,'full',v['full_mx'],'viscous',v['viscous_mx'],'numerical',v['numerical_mx'],flush=True)
(O/'manufactured-shear-summary.json').write_text(json.dumps({'definition':'Fixed analytic Spalding momentum profile and eddy viscosity; spatial refinement at a fixed physical face; not an SST manufactured full solution.','results':allrows,'commands':commands},indent=2))


