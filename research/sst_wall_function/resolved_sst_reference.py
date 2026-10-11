"""Independent 1D variable-density SST Couette FV reference.

No production SST/wall leaves are called. Four coupled FV equations (U,T,k,
omega) use positive implicit destruction/Picard iteration and a geometrically
refined physical-wall grid. The wall k is zero; omega uses the TMR finite wall
value 60 nu/(beta1*y1^2). This is an independent spatial discretization.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded

MU=2e-5; CP=1040.; GAS_R=8314.46261815324/28.; PRESSURE=101325.
L=.01; EDGE_U=30.; EDGE_T=500.; EDGE_K=.5; EDGE_W=200.

def solve(n=256, first=1e-7):
    y=np.r_[0.,np.geomspace(first,L,n-1)]; h=np.diff(y)
    vol=(y[2:]-y[:-2])/2
    U=EDGE_U*np.log1p(y/3e-5)/np.log1p(L/3e-5)
    theta=np.zeros(n); k=3*(1-np.exp(-y/2e-4)); k[-1]=EDGE_K
    omega=6*MU/(PRESSURE/(GAS_R*EDGE_T)*.075*np.maximum(y,first)**2)+EDGE_W
    omega[0]=60*MU/(PRESSURE/(GAS_R*EDGE_T)*.075*first**2);omega[-1]=EDGE_W
    def scalar(diff,positive,sink,left,right):
        a=(diff[1:]+diff[:-1])/(2*h)
        lower=a[:-1];upper=a[1:];diag=lower+upper+vol*sink[1:-1]
        rhs=vol*positive[1:-1];rhs[0]+=lower[0]*left;rhs[-1]+=upper[-1]*right
        ab=np.zeros((3,n-2));ab[0,1:]=-upper[:-1];ab[1]=diag;ab[2,:-1]=-lower[1:]
        return np.r_[left,solve_banded((1,1),ab,rhs,check_finite=False),right]
    history=[]
    for iteration in range(40000):
        T=EDGE_T+theta;rho=PRESSURE/(GAS_R*T);nu=MU/rho
        dU=np.gradient(U,y,edge_order=2);dk=np.gradient(k,y,edge_order=2);dw=np.gradient(omega,y,edge_order=2)
        ys=np.maximum(y,first/10); cd=2*.856*dk*dw/omega
        arg1=np.minimum(np.maximum(np.sqrt(k)/(.09*omega*ys),500*nu/(ys*ys*omega)),4*.856*k/(np.maximum(cd,1e-10)*ys*ys))
        f1=np.tanh(np.minimum(arg1,10)**4)
        arg2=np.maximum(2*np.sqrt(k)/(.09*omega*ys),500*nu/(ys*ys*omega));f2=np.tanh(np.minimum(arg2,100)**2)
        nut=.31*k/np.maximum(.31*omega,np.abs(dU)*f2)
        muEff=MU+rho*nut
        p=np.minimum(nut*dU*dU,.9*k*omega)
        beta=.075*f1+.0828*(1-f1);gamma=(5/9)*f1+.44*(1-f1)
        pw=gamma*np.minimum(dU*dU,(.9/.31)*omega*np.maximum(.31*omega,np.abs(dU)*f2))
        cross=(1-f1)*cd
        nextU=scalar(muEff,np.zeros(n),np.zeros(n),0,EDGE_U)
        # Viscous work converts mean kinetic energy to thermal energy.
        nextTheta=scalar(MU*CP/.72+rho*nut*CP/.9,muEff*dU*dU,np.zeros(n),0,0)
        nextK=scalar(MU+rho*(.85*f1+1*(1-f1))*nut,rho*p,rho*.09*omega,0,EDGE_K)
        nextW=scalar(MU+rho*(.5*f1+.856*(1-f1))*nut,rho*(pw+np.maximum(cross,0)),rho*(beta*omega+np.maximum(-cross,0)/omega),omega[0],EDGE_W)
        values=(U,theta,k,omega);nxt=(nextU,nextTheta,nextK,nextW)
        change=max(np.max(np.abs(b[1:-1]-a[1:-1])/np.maximum(np.abs(a[1:-1]),floor)) for a,b,floor in zip(values,nxt,(1.,1.,1e-10,1.)))
        relaxation=.5
        U,theta,k,omega=(a+relaxation*(b-a) for a,b in zip(values,nxt))
        if np.min(k)<0 or np.min(omega)<=0: raise RuntimeError('nonpositive iterate')
        if iteration%100==0:history.append([iteration,float(change)])
        if change<2e-10:break
    else:raise RuntimeError(f'not converged: {change}')
    # Re-evaluate the final-state nonlinear equations, independently of the
    # Picard update stopping test. Normalize each FV row by gross flux/source.
    T=EDGE_T+theta;rho=PRESSURE/(GAS_R*T);nu=MU/rho
    dU=np.gradient(U,y,edge_order=2);dk=np.gradient(k,y,edge_order=2);dw=np.gradient(omega,y,edge_order=2)
    ys=np.maximum(y,first/10);cd=2*.856*dk*dw/omega
    arg1=np.minimum(np.maximum(np.sqrt(k)/(.09*omega*ys),500*nu/(ys*ys*omega)),4*.856*k/(np.maximum(cd,1e-10)*ys*ys))
    f1=np.tanh(np.minimum(arg1,10)**4)
    f2=np.tanh(np.minimum(np.maximum(2*np.sqrt(k)/(.09*omega*ys),500*nu/(ys*ys*omega)),100)**2)
    nut=.31*k/np.maximum(.31*omega,np.abs(dU)*f2);muEff=MU+rho*nut
    p=np.minimum(nut*dU*dU,.9*k*omega)
    beta=.075*f1+.0828*(1-f1);gamma=(5/9)*f1+.44*(1-f1)
    pw=gamma*np.minimum(dU*dU,(.9/.31)*omega*np.maximum(.31*omega,np.abs(dU)*f2));cross=(1-f1)*cd
    def residual(field,diff,positive,sink):
        face=-(diff[1:]+diff[:-1])/(2*h)*np.diff(field)
        source=vol*(positive[1:-1]-sink[1:-1]*field[1:-1])
        defect=np.diff(face)-source
        scale=np.abs(face[:-1])+np.abs(face[1:])+vol*(np.abs(positive[1:-1])+np.abs(sink[1:-1]*field[1:-1]))
        return float(np.max(np.abs(defect)/np.maximum(scale,1e-30)))
    residuals={
        'U':residual(U,muEff,np.zeros(n),np.zeros(n)),
        'temperature':residual(theta,MU*CP/.72+rho*nut*CP/.9,muEff*dU*dU,np.zeros(n)),
        'k':residual(k,MU+rho*(.85*f1+1*(1-f1))*nut,rho*p,rho*.09*omega),
        'omega':residual(omega,MU+rho*(.5*f1+.856*(1-f1))*nut,rho*(pw+np.maximum(cross,0)),rho*(beta*omega+np.maximum(-cross,0)/omega))}
    tau=MU*U[1]/y[1];heat=-MU*CP/.72*(theta[1]-theta[0])/y[1]
    return dict(nodes=n,first=first,iterations=iteration+1,relative_update=float(change),normalized_discrete_residual=residuals,tau=float(tau),heat=float(heat),y=y.tolist(),U=U.tolist(),T=T.tolist(),k=k.tolist(),omega=omega.tolist(),history=history)

if __name__=='__main__':
    out=Path(__file__).parent/'results';out.mkdir(exist_ok=True)
    results=[]
    for n in [128,256,512,1024,2048,4096,8192]:
        r=solve(n);results.append(r);print({k:r[k] for k in ['nodes','iterations','relative_update','tau','heat']},flush=True)
    (out/'resolved_reference.json').write_text(json.dumps(results,indent=2))
    (out/'reference_8192.json').write_text(json.dumps(results[-1],indent=2))
    grid=[]
    for r in results:
        row={key:r[key] for key in ['nodes','iterations','relative_update','normalized_discrete_residual','tau','heat']}
        y=np.asarray(r['y']);rho=PRESSURE/(GAS_R*np.asarray(r['T']));averages=[]
        for yp in [.00075,.0015,.00225]:
            z=np.r_[0,y[(y>0)&(y<2*yp)],2*yp]
            averages.append(float(np.trapezoid(np.interp(z,y,rho*np.asarray(r['k'])),z)/np.trapezoid(np.interp(z,y,rho),z)))
        row['cell_average_k']=averages;grid.append(row)
    (out/'grid_convergence.json').write_text(json.dumps(grid,indent=2))
