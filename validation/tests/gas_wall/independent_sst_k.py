"""High precision, scalar SST k-equation oracle; no production kernels imported.

The fixture uses constant viscosity, equal species molar mass and no imposed
resolved tangential gradient. This is an independent evaluation/differentiation
of that equation, not a claim about the continuous BVP's discretization error.
Requires mpmath. Historical references remain in their original JSON file.
"""
import mpmath as mp

def derivative(capture, legacy_face_average=False):
    with mp.workdps(60):
        a=list(map(lambda x:mp.mpf(str(x)),capture['input']))
        L,_,wallT,_,_,edgeK,_,mass=a[:8]
        rows=[[mp.mpf(str(x)) for x in row] for row in capture['profile']]
        node=capture['node'];mu=mp.mpf('0.00002');pressure=mp.mpf(101325)
        gasR=mp.mpf('8.31446261815324')/mp.mpf('.028')
        betaStar=mp.mpf('.09');a1=mp.mpf('.31');alpha2=mp.mpf('.856')
        def properties(row):
            y,u,v,theta,Y,K,z=row
            return pressure/(gasR*(wallT+theta)),K*K,1/(z*z)
        def centered(left,value,right,h0,h1):
            return h1*(value-left)/(h0*(h0+h1))+h0*(right-value)/(h1*(h0+h1))
        def model(row,du,dv,dnormal,dk,domega):
            y=row[0];rho,k,omega=properties(row)
            cd=2*alpha2*dk*domega/omega
            viscous=500*mu/(rho*y*y*omega)
            f1=mp.tanh(min(max(mp.sqrt(k)/(betaStar*omega*y),viscous),4*alpha2*k/(max(cd,mp.mpf('1e-10'))*y*y),10)**4)
            f2=mp.tanh(min(max(2*mp.sqrt(k)/(betaStar*omega*y),viscous),100)**2)
            strain=du*du+dv*dv+2*dnormal*dnormal
            nut=a1*k/max(a1*omega,f2*mp.sqrt(strain))
            production=min(nut*(strain-mp.mpf(2)/3*dnormal*dnormal),10*betaStar*k*omega)
            return rho,k,omega,nut,1-mp.mpf('.15')*f1,rho*(production-mp.mpf(2)/3*dnormal*k-betaStar*k*omega)
        def residual(value):
            q=[r[:] for r in rows];q[node][5]=value
            def flux(l,r):
                x,y=q[l],q[r];h=y[0]-x[0];px,py=properties(x),properties(y)
                normal=mass*gasR/pressure*(y[3]-x[3])/h
                grad=((y[1]-x[1])/h,(y[2]-x[2])/h,normal,(py[1]-px[1])/h,(py[2]-px[2])/h)
                left=model(x,*grad);right=model(y,*grad)
                if legacy_face_average:
                    conductance=mu+(left[0]+right[0])*(left[3]+right[3])*(left[4]+right[4])/8
                else:conductance=mu+(left[0]*left[3]*left[4]+right[0]*right[3]*right[4])/2
                return mass*(px[1]+py[1])/2-conductance*(py[1]-px[1])/h
            l,p,r=q[node-1],q[node],q[node+1];h0=p[0]-l[0];h1=r[0]-p[0]
            pl,pp,pr=properties(l),properties(p),properties(r)
            du=centered(l[1],p[1],r[1],h0,h1);dv=centered(l[2],p[2],r[2],h0,h1)
            dn=mass*gasR/pressure*centered(l[3],p[3],r[3],h0,h1)
            dk=centered(pl[1],pp[1],pr[1],h0,h1);dw=centered(pl[2],pp[2],pr[2],h0,h1)
            source=model(p,du,dv,dn,dk,dw)[-1]
            scale=(abs(mass)+mu/L)*max(1,edgeK)
            return (flux(node,node+1)-flux(node-1,node)-(h0+h1)*source/2)/scale
        return float(mp.diff(residual,rows[node][5]))
