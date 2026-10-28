#!/usr/bin/env python3
"""Independent original equations; consumes actual CUDA constitutive results."""
import json,math,sys
from pathlib import Path
rows=Path(sys.argv[1]);errors={k:0. for k in ('mu','kappa','D','diffusive_flux','enthalpy_flux','wall_energy')}
count=0
for line in rows.read_text().splitlines():
 T,p,y,mu,k,D,j0,j1,h,wm,wp,wd,wh,wj0,wj1=map(float,line.split())
 mu_ref=(17.10-4.65*y+1.20*y*y)*1e-6*(T/273.)**1.5*(273.+114.+220.*y)/(T+114.+220.*y)
 pr=.720+.064*y;cp=(1-y)*1004.675+y*819.27;k_ref=mu_ref*cp/pr
 D_ref=5.75e-10*T**1.81*101325./p
 R=8.31446261815324*((1-y)/.02896520682164515+y/.0440095);rho=p/(R*T)
 j_ref=rho*D_ref*.7;h_ref=j_ref*(1004.675-819.27)*T
 for key,a,b in [('mu',mu,mu_ref),('kappa',k,k_ref),('D',D,D_ref),('diffusive_flux',j0,j_ref),('enthalpy_flux',h,h_ref),('wall_energy',wh,h_ref-2*k_ref)]:
  err=abs(a-b)/max(abs(b),1e-12);errors[key]=max(errors[key],err);assert math.isfinite(a) and err<3e-13,(key,T,p,y,a,b,err)
 assert j0+j1==0 and wj0+wj1==0
 assert abs(wm-mu_ref)<3e-13*mu_ref and abs(wp-pr)<1e-14 and abs(wd-D_ref)<3e-13*D_ref
 assert abs(wj0-j_ref)<3e-13*j_ref
 count+=1
assert count==126
print(json.dumps(dict(samples=count,maximumRelativeError=errors,correctedSpeciesSum=0),indent=2))
