#include "ablation/InterfaceMath.H"
#include <fenv.h>
#include <iostream>
#include <iomanip>
using namespace chmt;
int main(){feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);std::cout<<std::setprecision(17);PhysicsConfig p;double b[Ns]{},j[Ns]{},y[Ns]{},f[Ns]{};
if(Ns==2){p.species[0].R=100;p.species[1].R=200;p.gasDiffusivity[0]=p.gasDiffusivity[1]=1;b[0]=j[0]=1;bool ok=interfaceComposition(j,b,1,1,1,p,1e-12,y);std::cout<<ok<<' '<<y[0]<<' '<<y[1]<<'\n';return !(ok&&y[0]==1&&y[1]==0);}
if(Ns==3){double D[]={1,2,3},bulk[]={1./3,1./3,1./3},Y[]={.6,.1,.3};double S=0;for(int i=0;i<Ns;++i){p.species[i].R=1;p.gasDiffusivity[i]=D[i];b[i]=bulk[i];S+=D[i]*(Y[i]-b[i]);}double J=-2.8;for(int i=0;i<Ns;++i)j[i]=J*Y[i]+D[i]*(Y[i]-b[i])-Y[i]*S;composition_detail::Info info;bool ok=composition_detail::scalarSolve(j,b,300,300,1,p,1e-10,y,&info);compositionResidual(y,b,j,300,300,1,p,f,nullptr);std::cout<<"physical "<<ok<<" ambiguous "<<info.ambiguous<<" state "<<int(info.state)<<" residual "<<info.actualResidual<<'\n';for(int i=0;i<Ns;++i)std::cout<<y[i]<<' '<<f[i]<<'\n';
double left[]={-.5,-.5,-.2},right[]={.5,.5,.8};auto poly=composition_detail::product(left,right);composition_detail::Info pi;double roots[composition_detail::ScalarCandidateCapacity]{};int n=composition_detail::roots(poly,pi,roots);std::cout<<"polynomial "<<n;for(int i=0;i<n;++i)std::cout<<' '<<roots[i];std::cout<<'\n';bool found=false;for(int i=0;i<n;++i)found=found||std::abs(roots[i]-.5)<1e-12;return !(ok&&found);}}
