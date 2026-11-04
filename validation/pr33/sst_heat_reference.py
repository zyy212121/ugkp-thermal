#!/usr/bin/env python3
"""Independent NumPy FV equations and SciPy nonlinear solve for captured SST.

This restricted reference evaluates the supplied pure-gas, zero-blowing fixture.
It imports no production residual, Jacobian, SST or transport functions. A
perturbed supplied iterate initializes SciPy's independent dense trust-region
solve. This certifies the discrete equation, not experimental model accuracy.
Dependencies: numpy, scipy; input is sst_profile.cpp JSON on standard input.
"""
import sys,json
import numpy as np
from scipy.optimize import least_squares
P=120497.45682037952;TW=290.53216418474562;TE=283.55161010181268
MU=1.7244421537657099e-5;CP=1004.675;PR=.72;PRT=.9
RGAS=8.31446261815324/.028965206821645145
UE=112.41422500408262;KE=8.946693975448385;WE=123066.88751570872

def model(y,rho,k,z,du,dk,dw):
    omega=1/z**2
    cd=2*.856*dk*dw/omega
    visc=500*MU/(rho*y*y*omega)
    f1=np.tanh(np.minimum(10,np.minimum(np.maximum(np.sqrt(k)/(.09*omega*y),visc),4*.856*k/(np.maximum(cd,1e-10)*y*y)))**4)
    f2=np.tanh(np.minimum(100,np.maximum(2*np.sqrt(k)/(.09*omega*y),visc))**2)
    nut=.31*k/np.maximum(.31*omega,f2*np.abs(du))
    sk=rho*(np.minimum(nut*du**2,.9*k*omega)-.09*k*omega)
    production=(.44+(5/9-.44)*f1)*np.minimum(du**2,(10/.31)*.09*omega*np.maximum(.31*omega,f2*np.abs(du)))
    return nut,f1,sk,production

def logmean(a,b):
    ratio=(b-a)/a
    out=np.array(a,copy=True);active=ratio!=0
    out[active]=(b[active]-a[active])/np.log1p(ratio[active]);return out

def work_weight(ma,mb,ba,bb,quadrature=2):
    # Integrate u/B in the logarithmic momentum-resistance coordinate.
    reverse=mb<ma
    l=np.where(reverse,mb,ma);r=np.where(reverse,ma,mb)
    bl=np.where(reverse,bb,ba);br=np.where(reverse,ba,bb)
    change=r-l;active=change!=0;answer=np.full(len(ma),.5)
    nodes,weights=np.polynomial.legendre.leggauss(quadrature);nodes=(nodes+1)/2
    t=nodes[:,None];lm=np.log(r[active]/l[active]);local=l[active]*np.exp(t*lm)
    position=(local-l[active])/change[active]
    conductivity=bl[active]+position*(br[active]-bl[active])
    density=weights[:,None]*local/conductivity
    answer[active]=(density*t).sum(axis=0)/density.sum(axis=0)
    return np.where(reverse,1-answer,answer)

class Reference:
    def __init__(self,data,quadrature=2):
        self.y=np.array(data['nodes'])[:,0];self.n=len(self.y);self.quadrature=quadrature
        # Only u, theta, sqrt(k), z are active for this pure-gas fixture.
        self.base=np.array(data['nodes'])[:,[1,3,5,6]]
        self.base[0]=0;self.base[-1]=[UE,TE-TW,np.sqrt(KE),1/np.sqrt(WE)]
        self.base[1,3]=self.y[1]*np.sqrt((P/(RGAS*TW))*.075/(6*MU))
        self.mask=np.ones((self.n,4),dtype=bool);self.mask[[0,-1]]=False;self.mask[1,3]=False
        self.variable_scale=np.array([UE,10,np.sqrt(KE),1/np.sqrt(WE)])
    def unpack(self,x):
        q=self.base.copy();q[self.mask]=(x*self.variable_scale[np.nonzero(self.mask)[1]])
        return q
    def pack(self,q):return (q/self.variable_scale)[self.mask]
    def evaluate(self,q):
        y=self.y;dy=np.diff(y);u,theta,K,z=q.T;k=K*K;rho=P/(RGAS*(TW+theta))
        omega=np.zeros(self.n);omega[1:]=1/z[1:]**2
        du=np.diff(u)/dy;dk=np.diff(k)/dy;dw=np.diff(omega)/dy
        # First interval uses the encoded singular wall omega asymptote.
        dw_first=-2*(z[1]-z[0])/dy[0]/z[1]**3
        nur,f1r,_,_=model(y[1:],rho[1:],k[1:],z[1:],du,dk,np.r_[dw_first,dw[1:]])
        nul=np.zeros(self.n-1);f1l=np.zeros(self.n-1)
        nul[1:],f1l[1:],_,_=model(y[1:-1],rho[1:-1],k[1:-1],z[1:-1],du[1:],dk[1:],dw[1:])
        ma=MU+rho[:-1]*nul;mb=MU+rho[1:]*nur
        ba=MU/PR+rho[:-1]*nul/PRT;bb=MU/PR+rho[1:]*nur/PRT
        tau=logmean(ma,mb)*du
        heat=-CP*logmean(ba,bb)*np.diff(theta)/dy-tau*(u[:-1]+work_weight(ma,mb,ba,bb,self.quadrature)*np.diff(u))
        Dk=MU+(rho[:-1]*(1-.15*f1l)*nul+rho[1:]*(1-.15*f1r)*nur)/2
        gradient_k=dk.copy();power=(1+np.sqrt(1+24*.09/.075))/2
        gradient_k[0]=k[1]*power*.5**(power-1)/y[1]
        fluxK=-Dk*gradient_k
        Dw=MU+(rho[:-1]*(.856+(.5-.856)*f1l)*nul+rho[1:]*(.856+(.5-.856)*f1r)*nur)/2
        fit=np.ones(self.n-1);fit[1:]=(y[1:-1]*y[2:]/((y[1:-1]+y[2:])/2)**2)**2
        fluxW=-Dw*fit*dw
        h0=dy[:-1];h1=dy[1:];volume=(h0+h1)/2
        def centered(v):return h1*np.diff(v)[:-1]/(h0*(h0+h1))+h0*np.diff(v)[1:]/(h1*(h0+h1))
        gu=centered(u);gk=centered(k);gw=centered(omega)
        gw[0]=-2*centered(z)[0]/z[1]**3
        nut,f1,sourceK,production=model(y[1:-1],rho[1:-1],k[1:-1],z[1:-1],gu,gk,gw)
        row=np.zeros_like(q)
        L=y[-1];row[1:-1,0]=(-np.diff(tau))/(MU*UE/L)
        row[1:-1,1]=np.diff(heat)/(MU*CP/PR/L*TW)
        row[1:-1,2]=(np.diff(fluxK)-volume*sourceK)/(MU/L*max(1,KE))
        # Integrate the local omega^2 asymptote analytically, and direct the
        # cross-diffusion logarithmic gradient along its transport sign.
        lo=(y[:-2]+y[1:-1])/2;hi=(y[1:-1]+y[2:])/2
        square_mean=omega[1:-1]**2*y[1:-1]**4*(lo**-3-hi**-3)/(3*volume)
        loggrad=np.zeros(self.n-2)
        negative=gk<0;positive=gk>0
        valid=negative.copy();valid[0]=False
        loggrad[valid]=np.log(omega[1:-1][valid]/omega[:-2][valid])/h0[valid]
        loggrad[positive]=np.log(omega[2:][positive]/omega[1:-1][positive])/h1[positive]
        sourceW=rho[1:-1]*(production-(.0828+(.075-.0828)*f1)*square_mean+2*.856*(1-f1)*gk*loggrad)
        row[1:-1,3]=-(np.diff(fluxW)-volume*sourceW)*z[1:-1]**4/(2*volume*(P/(RGAS*TE)))
        return row[self.mask],heat
    def residual(self,x):return self.evaluate(self.unpack(x))[0]

def run(data,perturbation=.01):
    ref=Reference(data);initial=ref.pack(ref.base)
    defect=float(np.max(np.abs(ref.residual(initial))))
    x=initial*(1+perturbation*np.sin(np.arange(len(initial))))
    # Independent solver, Jacobian and perturbation sequence.
    solved=least_squares(ref.residual,x,jac='3-point',xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=300,x_scale='jac')
    state=ref.unpack(solved.x);r,flux=ref.evaluate(state)
    heat=float(np.mean(flux));error=abs(heat-data['heat'])
    assert solved.success and max(abs(r))<1e-8,(solved.message,max(abs(r)))
    assert defect<1e-8 and error<1e-5,(defect,error)
    refined=Reference(data,64);_,flux64=refined.evaluate(state)
    return {'nodes':ref.n,'relative_initial_perturbation':perturbation,'production_heat':data['heat'],'independent_heat':heat,'heat_difference':error,'production_equation_defect':defect,'independent_residual':float(max(abs(r))),'scipy_evaluations':solved.nfev,'higher_order_work_quadrature_max_flux_difference':float(max(abs(flux64-flux)))}
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--perturbation',type=float,default=.01)
    args=parser.parse_args();print(json.dumps(run(json.load(sys.stdin),args.perturbation),indent=2))
