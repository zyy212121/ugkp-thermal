#include <initializer_list>
#include <cstdio>
#include <cmath>
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
template<class Storage> void run(){
 ugkwp::ParticleCaloric law;
 const double rows[][4]={
 {3917.331787109375,4.4000000000000002e-7,.63645399999999996,1506.7985147669101},
 {834.5579833984375,-2.6e-7,.79153200000000001,3339.4206502848047},
 {3780.924072265625,3.7e-7,.108899,3588.2611494543198},
 {834.55799999999999,-2.6e-7,.79153200000000001,3339.4206371441369},
 {790.82860000000005,-7.0999999999999998e-7,.33108700000000002,-61.833068688043909},
 {1006.1807,5.8999999999999996e-7,.70327499999999998,3292.3782682072242},
 {3000,.001,.8,200},{3000,.001,1,0},{200,0,1,0},{4000,0,1,0},
 {3000,0,0,200},{3000,0,0,4000}};
 for(const auto& row:rows){
  ugkwp::ParticleThermalCarry<Storage> p{Storage(row[0]),row[1]};
  const double t=ugkwp::effectiveParticleTemperature(p,law);
  const double next=t+std::fma(row[2]-1,t,row[3]);
  // A row rounded to the other storage precision may be old-refused.
  if(next<200||next>4000)continue;
  auto r=ugkwp::publishAffineRadiationThermalHeat(p,row[2],row[3],Storage(200),Storage(4000),law);
  std::printf("affine %zu %a %d %d %a %a %a\n",sizeof(Storage),row[0],r.valid,r.limited,double(p.temperature),p.specificEnthalpyResidual,r.acceptedSpecificHeat);
 }
 for(double t:{200.,3000.,4000.})for(double carry:{-1.e-12,0.,1.e-12})for(double q:{-1.e12,-.000001,0.,.000001,1.e12}){
  ugkwp::ParticleThermalCarry<Storage> p{Storage(t),carry};
  auto r=ugkwp::publishParticleThermalHeat(p,q,Storage(200),Storage(4000),law);
  std::printf("heat %zu %a %a %a %d %d %a %a %a\n",sizeof(Storage),t,carry,q,r.valid,r.limited,double(p.temperature),p.specificEnthalpyResidual,r.acceptedSpecificHeat);
 }
 for(double t:{200.,3000.,4000.})for(double wall:{300.,5000.})for(double integral:{0.,1.e-12,1.}){
  ugkwp::ParticleThermalCarry<Storage> p{Storage(t),0.};
  const auto r=ugkwp::publishFiniteWallThermalHeat(p,wall,1.e-6,100.,integral,Storage(200),Storage(4000),law);
  std::printf("wall %zu %a %a %a %d %a %a %a\n",sizeof(Storage),t,wall,integral,r.valid,double(p.temperature),p.specificEnthalpyResidual,r.wallEnergy);
 }
}
int main(){run<double>();}
