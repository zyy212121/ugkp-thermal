#include "materials/MaterialCaloric.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
PhysicsConfig physics() {
    PhysicsConfig p; p.cfl=.4;
    for(int c=0;c<Nc;++c) { auto& x=p.condensed[c]; x.rho=1000; x.cp0=900+100*c; x.cp1=.1; x.e0=-1e8*c; x.Tmin=100; x.Tmax=3000; x.conductivity=2; }
    for(int s=0;s<Ns;++s) { auto& x=p.species[s]; x.R=287+10*s; x.cp0=1100; x.cp1=.2; x.e0=-1e7*s; x.Tmin=100; x.Tmax=3000; }
    return p;
}
SolidQ solid(const PhysicsConfig& p,double temperature=600,double pore=1e-4) {
    SolidQ q; q.condensed[0]=.9; q.pore[0]=pore; q.porosity=.1;
    q.energy=q.condensed[0]*condensedE(p.condensed[0],temperature)+pore*speciesE(p.species[0],temperature); return q;
}
int main() {
    auto p=physics(); auto q=solid(p); MaterialCaloric caloric; std::string error;
    assert(materialCaloric(q,p,caloric,error));
    const double t0=600,t1=900;
    assert(std::abs((caloric.energy(t1)-caloric.energy(t0))-caloric.secant(t1,t0)*(t1-t0))<1e-7);
    q.condensed[0]=.4; q.condensed[1]=.5;
    assert(materialCaloric(q,p,caloric,error));
    assert(caloric.energy(600)<0); // Formation-inclusive U need not be positive.
    const double expected=.4*condensedE(p.condensed[0],600)+.5*condensedE(p.condensed[1],600)+q.pore[0]*speciesE(p.species[0],600);
    assert(caloric.energy(600)==expected);
    q.condensed[0]=-1; assert(!materialCaloric(q,p,caloric,error));
    std::cout<<"CPU material caloric tests passed\n";
}
