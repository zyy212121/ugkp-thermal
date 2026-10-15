#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cfenv>
unsigned long long adapterConstructions=0;
#include "ADAPTER_HEADER"
#include "materials/MaterialCaloric.H"
using namespace chmt;
void require(bool ok){if(!ok)std::abort();}
int main(){std::cout<<std::hexfloat;
 // Original valid/inactive-species routes must also work with floating traps.
 #if defined(__GLIBC__)
 feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
 #endif
 const Real intervals[][2]={{150,400},{1100,2200},{800,1200},{1200,800},{1000,1000},{500,500},{1000,1200},{800,1000}};
 for(int model=0;model<2;++model)for(Real formation:{0.,1e12,1e120})for(const auto& interval:intervals){
  SpeciesThermo s;s.R=287;s.cp0=1100;s.cp1=.17;s.e0=formation;s.Tmin=100;s.Tmid=1000;s.Tmax=3000;
  if(model){s.model=ugkwp::SpeciesThermoModel::NASA7;
   const Real low[7]={3.6,1e-4,1e-8,-1e-12,1e-16,formation,2};
   const Real high[7]={3.8,2e-4,-1e-8,2e-12,-1e-16,formation+250,3};
   for(int j=0;j<7;++j){s.nasa7[j]=low[j];s.nasa7[j+7]=high[j];}}
  const Real a=interval[0],b=interval[1];adapterConstructions=0;
  const Real secant=speciesEnergySecant(s,a,b);const auto count=adapterConstructions;
  require(chmt::finite(secant));std::cout<<secant<<' '<<speciesCv(s,a)<<' '<<speciesCv(s,b)<<'\n';
  std::cerr<<model<<' '<<(a==b)<<' '<<(model&&minValue(a,b)<=s.Tmid&&maxValue(a,b)>s.Tmid)<<' '<<count<<'\n';
  MaterialCaloric c;c.a0=formation;c.a1=2300;c.a2=.031;c.gas[0]=s;c.pore[0]=.35;
  // All other gas tables remain unconfigured and inactive, preserving the real consumer.
  std::cout<<c.secant(a,b)<<' '<<c.capacity(a)<<' '<<c.energy(a)<<'\n';
 }
 return 0;}
