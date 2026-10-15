#include "OpenFoamWallFunctions.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
int main() {
 int count=0; double worst=0;
 for(double rho:{0.5,3.0,30.0}) for(double tk:{0.0,0.01,10.0,200.0,30000.0})
 for(double y:{1e-6,1e-4,1e-3}) for(double velocity:{0.0,30.0,300.0,1500.0})
 for(double dt:{-2700.0,-100.0,-1e-8,0.0,1e-8,100.0,2700.0}) {
 const GpuReal cp=1722,mu=9.29152148664344e-5,pr=0.4,prt=0.85,cmu=0.09,kap=0.41,E=9.8;
 const auto P=ugkpwall::jayatillekeSmoothP(pr/prt),yt=ugkpwall::jayatillekeThermalYPlus(pr/prt,kap,E);
 const GpuReal grad=dt/y;
 auto a=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,pr,prt,cmu,kap,E,P,yt,tk,y,velocity,0,grad);
 if(!a.valid||!std::isfinite(a.heatFlux))return 1;
 const double molecular=double(mu)*double(cp)/double(pr);
 const double u=std::pow(double(cmu),0.25)*std::sqrt(tk),yp=u*y*rho/double(mu);
 double kref=molecular,energy=0;
 if(tk>0) {
  const double tp=yp<yt?double(pr)*yp:double(prt)*(std::log(double(E)*yp)/double(kap)+double(P));
  const double uc=u/double(kap)*std::log(double(E)*double(yt));
  const double C=0.5*rho*u*(yp<yt?double(pr)*velocity*velocity:double(prt)*velocity*velocity+(double(pr)-double(prt))*uc*uc);
  kref=std::max(molecular,double(cp)*rho*u*y/tp);energy=C/tp;
 }
 double ref=-kref*double(grad)+energy;
 double scale=std::max(1.,std::abs(kref*double(grad))+std::abs(energy));
 double error=std::abs(a.heatFlux-ref)/scale;worst=std::max(worst,error);
 if(error>5e-6||std::abs(a.conductivity/kref-1)>3e-6)return 2;
 auto zero=ugkpwall::sstJayatillekeThermalTransport(rho,cp,mu,pr,prt,cmu,kap,E,P,yt,tk,y,velocity,0,0);
 if(!zero.valid||a.conductivity!=zero.conductivity)return 3;
 ++count;
 }
 std::cout<<"PASS "<<count<<" thermal energy-law cases; worst relative residual="<<worst<<"\n";
}
