#ifndef CHMT_TEST_SUPPORT_H
#define CHMT_TEST_SUPPORT_H
#include "materials/Thermodynamics.H"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
namespace chmt_test {
inline void check(bool ok,const std::string& what){if(!ok){std::cerr<<"FAIL: "<<what<<'\n';std::exit(1);}}
inline void near(double got,double want,const std::string& what,double relative=2e-11,double absolute=1e-12){check(std::abs(got-want)<=absolute+relative*std::max(std::abs(got),std::abs(want)),what);}
inline chmt::PhysicsConfig physics(){
 chmt::PhysicsConfig p;p.nElements=1;p.minDt=1e-12;p.maxDt=1;p.cfl=.4;
 for(int s=0;s<chmt::Ns;++s){auto& x=p.species[s];x.R=280+40*s;x.cp0=1100+100*s;x.cp1=.15+.02*s;x.e0=-200000+30000*s;x.Tmin=100;x.Tmax=2000;x.element[0]=1;}
 for(int c=0;c<chmt::Nc;++c){auto& x=p.condensed[c];x.rho=1000+200*c;x.cp0=900+150*c;x.cp1=.3;x.e0=-100000+30000*c;x.conductivity=2+c;x.Tmin=100;x.Tmax=2000;x.element[0]=1;}
 p.liquid=p.condensed[0];p.liquidReferencePressure=100000;p.liquidViscosity=.1;return p;
}
inline chmt::GasPrimitive gas(const chmt::PhysicsConfig& p){
 chmt::GasPrimitive w;w.rho=1.3;w.temperature=730;w.velocity={12,-4,2};w.Y[0]=.7;w.Y[1]=.3;
 chmt::GasPrimitive recovered;check(chmt::recoverGas(chmt::conservativeGas(w,1,p),1,p,recovered),"gas fixture EOS");return recovered;
}
}
#endif
