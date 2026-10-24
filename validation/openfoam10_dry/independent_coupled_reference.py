"""Independent SciPy method-of-lines reference of the declared synthetic slab.
No OpenFOAM or CHMT header imported. BDF integrates unsplit extensive balances;
temperature is independently eliminated by the constant-cv caloric formula.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
import json
ROOT=Path(__file__).resolve().parent

def reference(n):
 dx=1/n;x=(np.arange(n)+.5)*dx
 m=np.full(n,500.);g0=.5*(.6+.1*np.sin(2*np.pi*x));g1=.5-g0
 E=m*700000+g0*410000+g1*310000
 q0=np.column_stack((m,g0,g1,E)).ravel()
 def primitive(q):
  m,a,b,E=q.T;T=(E-400000*m-200000*a-100000*b)/(1000*m+700*(a+b));phi=1-m/1000;rho=(a+b)/phi;p=rho*300*T
  return T,phi,rho,p,a/(a+b),10*m
 def rhs(t,q):
  q=q.reshape(n,4);m,a,b,E=q.T;T,phi,rho,p,Y,k=primitive(q)
  Tf=np.r_[340,T,300];pf=np.r_[90000,p,90000];rf=np.r_[90000/(300*340),rho,1];yf=np.r_[.6,Y,.6]
  dist=np.full(n+1,dx);dist[[0,-1]]=dx/2
  u=-1e-5*np.diff(pf)/dist
  donor=np.where(u>=0,np.arange(n+1),np.arange(1,n+2))
  adv=u*rf[donor];j=-.003*.5*(rf[:-1]+rf[1:])*np.diff(yf)/dist
  f0=adv*yf[donor]+j;f1=adv*(1-yf[donor])-j
  # h0=200000+1000T, h1=100000+1000T, diffusion h0-h1=100000.
  fe=adv*(100000+100000*yf[donor]+1000*Tf[donor])+100000*j
  kface=np.r_[k[0],.5*(k[:-1]+k[1:]),k[-1]]
  fe-=kface*np.diff(Tf)/dist
  r=.4*np.exp(2000*(1/300-1/T))*m
  return np.column_stack((-r,.6*r-np.diff(f0)/dx,.4*r-np.diff(f1)/dx,-np.diff(fe)/dx)).ravel()
 sparsity=lil_matrix((4*n,4*n))
 for i in range(n):sparsity[4*i:4*i+4,4*max(0,i-1):4*min(n,i+2)]=1
 sol=solve_ivp(rhs,(0,.02),q0,method='BDF',rtol=2e-10,atol=np.tile([1e-9,1e-12,1e-12,1e-4],n),jac_sparsity=sparsity.tocsr(),max_step=.0002)
 if not sol.success:raise RuntimeError(sol.message)
 final=sol.y[:,-1].reshape(n,4);T,phi,rho,p,Y,k=primitive(final)
 data=np.column_stack((x,T,p,phi,final[:,0],Y))
 np.savetxt(ROOT/f'results/independent_coupled_N{n}.csv',data,delimiter=',',header='x,T,p,phi,solidDensity,Y0',comments='')
 return {'N':n,'nfev':sol.nfev,'njev':sol.njev,'steps':len(sol.t),'Tmin':T.min(),'Tmax':T.max()}
if __name__=='__main__':
 results=[reference(n) for n in [20,40,80,160]]
 (ROOT/'results/independent_reference_summary.json').write_text(json.dumps(results,indent=2));print(results)
