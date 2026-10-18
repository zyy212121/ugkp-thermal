#include "ablation/InterfaceMath.H"
#include <fenv.h>
#include <iostream>
#include <iomanip>
using namespace chmt;
int main(){feenableexcept(FE_OVERFLOW|FE_INVALID|FE_DIVBYZERO);std::cout<<std::setprecision(17)<<std::unitbuf;PhysicsConfig p;Real d[Ns]{},u[Ns]{},v[Ns]{},r[Ns]{},bulk[Ns]{},out[Ns]{},Js[Ns]{};for(int i=0;i<Ns;++i){r[i]=p.species[i].R=1;}d[1]=d[2]=d[3]=p.gasDiffusivity[1]=p.gasDiffusivity[2]=p.gasDiffusivity[3]=1e-308;u[1]=Js[1]=1.5;u[2]=Js[2]=-1.5;u[3]=Js[3]=1.5;r[1]=p.species[1].R=r[3]=p.species[3].R=1e-310;v[2]=5e-309;bulk[0]=bulk[2]=.5;std::cout<<"public finite inputs\n";bool ok=interfaceComposition(Js,bulk,1,1,1,p,1e-12,out);std::cout<<"public="<<ok<<"\n";std::cout<<"pole helper\n";ok=composition_detail::recoverPole(0,d,u,v,r,bulk,1e-310,1,out);std::cout<<"private="<<ok<<"\n";}