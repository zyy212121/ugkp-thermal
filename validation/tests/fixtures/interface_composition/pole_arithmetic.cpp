#include "ablation/InterfaceMath.H"
#include <fenv.h>
#include <iostream>
using namespace chmt;
int main(){
    const int mask=FE_OVERFLOW|FE_INVALID|FE_DIVBYZERO;
    feenableexcept(mask); PhysicsConfig p; Real d[Ns]{},u[Ns]{},v[Ns]{},r[Ns]{},bulk[Ns]{},out[Ns]{},Js[Ns]{};
    for(int i=0;i<Ns;++i){r[i]=p.species[i].R=1;out[i]=123;}
    d[1]=d[2]=d[3]=p.gasDiffusivity[1]=p.gasDiffusivity[2]=p.gasDiffusivity[3]=1e-308;
    u[1]=Js[1]=1.5;u[2]=Js[2]=-1.5;u[3]=Js[3]=1.5;
    r[1]=p.species[1].R=r[3]=p.species[3].R=1e-310;
    v[2]=5e-309;bulk[0]=bulk[2]=.5;
    composition_detail::Info info;
    bool ok=composition_detail::scalarSolve(Js,bulk,1,1,1,p,1e-12,out,&info);
    if(!ok){for(int i=0;i<Ns;++i)if(out[i]!=123)return 1;}
    else {Real F[Ns]{};if(!compositionResidual(out,bulk,Js,1,1,1,p,F,nullptr))return 2;
        Real sum=0;for(int i=0;i<Ns;++i){if(out[i]<0||out[i]>1)return 3;sum+=out[i];}
        if(absValue(sum-1)>1e-12)return 4;}
    bool direct=composition_detail::recoverPole(0,d,u,v,r,bulk,1e-310,1,out);
    // This input is not asserted to have a physical root. Neither mathematical
    // trial may trap, change the host trap mask, or publish a default on failure.
    if((fegetexcept()&mask)!=mask || fetestexcept(mask))return 5;
    std::cout<<"public="<<ok<<" direct="<<direct<<" typed="<<int(info.state)<<"\n";
}
