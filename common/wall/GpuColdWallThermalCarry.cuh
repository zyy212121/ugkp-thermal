#pragma once
#include "GpuParticleCaloric.H"
#include "GpuColdWallSolidification.H"
#include <cfloat>
#include <cmath>
#if defined(__CUDACC__)
#define UGKWP_COLD_CARRY_HD __host__ __device__
#else
#define UGKWP_COLD_CARRY_HD
#endif
namespace ugkwp {
struct ColdWallHeatPublication { bool valid; double acceptedSpecificHeat; };
// Equal-mass nodes: the one persistent scalar is a uniform specific-enthalpy
// correction, never a second copy of the profile or an unbounded heat ledger.
UGKWP_COLD_CARRY_HD inline float coldWallStoredEnthalpyFloor(double target) {

 if(target>double(FLT_MAX))return FLT_MAX;
 if(target<-double(FLT_MAX))return -FLT_MAX;
 float stored=static_cast<float>(target);
 if(double(stored)>target)stored=::nextafterf(stored,-INFINITY);
 return stored;
}
// Leave only the double-arithmetic cancellation margin before crediting a
// uniform cooling proposal. Float rounding is handled by directed publication.
UGKWP_COLD_CARRY_HD inline double coldWallUniformCoolingBound
(const float* nodes,int count,double residual) {
 double minimum=double(nodes[0])+residual,maximum=minimum;
 for(int j=1;j<count;++j){const double h=double(nodes[j])+residual;
  if(h<minimum)minimum=h;
  if(h>maximum)maximum=h;
 }
 const double margin=16.*0x1p-52*maximum+double(FLT_MIN);
 return minimum>margin?margin-minimum:0.;
}
UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication publishColdWallUniformSpecificHeat
(float* nodes, int count, double& residual, double specificHeat) {
 if (!nodes || count<=0 || !(::fabs(residual)<=DBL_MAX)
     || !(::fabs(specificHeat)<=DBL_MAX)) return {false,0.0};
 if (specificHeat==0.0) {
  for(int j=0;j<count;++j) {
   const double effective=double(nodes[j])+residual;
   if(!(::fabs(effective)<=DBL_MAX)) return {false,0.0};
  }
  return {true,0.0};
 }
 const double correction=residual+specificHeat;
 double storageChange=0.0;
 for(int j=0;j<count;++j) {
  const double effective=double(nodes[j])+residual;
  if(!(::fabs(effective)<=DBL_MAX)) return {false,0.0};
  const double target=double(nodes[j])+correction;
  const float stored=coldWallStoredEnthalpyFloor(target);
  if(!(::fabs(target)<=DBL_MAX) || !(::fabs(stored)<=FLT_MAX)) return {false,0.0};
  storageChange+=(double(stored)-double(nodes[j]))/count;
 }
 const double next=(residual-storageChange)+specificHeat;
 const double accepted=storageChange+(next-residual);
 if(!(::fabs(next)<=DBL_MAX) || !(::fabs(accepted)<=DBL_MAX))return {false,0.};
 for(int j=0;j<count;++j)nodes[j]=coldWallStoredEnthalpyFloor(double(nodes[j])+correction);
 residual=next;
 return {true,accepted};
}
template<class Parameters>
UGKWP_COLD_CARRY_HD inline ParticleCaloric coldWallCarryCaloric(const Parameters& p) {
 ParticleCaloric c;c.phaseChange=true;c.meltingTemperature=p.meltingTemperatureK;
 c.mushyRange=p.mushyRangeK;c.latentHeat=p.latentHeatJkg;c.solidCp=p.solidSpecificHeatJkgK;
 return c;
}
template<class Parameters>
UGKWP_COLD_CARRY_HD inline double coldWallCarrySolidFraction(double temperature,const Parameters& p) {
 const double tl=double(p.meltingTemperatureK)+.5*double(p.mushyRangeK);
 const double f=(tl-temperature)/double(p.mushyRangeK);
 return f<0.0?0.0:(f>1.0?1.0:f);
}
}
#ifdef GPU_PAIRED_PARTICLE_HEAT
using ColdWallThermalReal=double;
#else
using ColdWallThermalReal=GpuReal;
#endif
UGKWP_COLD_CARRY_HD inline ColdWallThermalReal coldWallCarryTemperature
(ColdWallThermalReal h,const Foam::gpuThermal::ColdWallSolidificationParameters& p) {
#ifdef GPU_PAIRED_PARTICLE_HEAT
 return ugkwp::coldWallCarryCaloric(p).temperature(h);
#else
 return Foam::gpuThermal::coldWallTemperatureFromSpecificEnthalpyK(h,p);
#endif
}
UGKWP_COLD_CARRY_HD inline ColdWallThermalReal coldWallCarrySpecificHeat
(ColdWallThermalReal t,const Foam::gpuThermal::ColdWallSolidificationParameters& p) {
#ifdef GPU_PAIRED_PARTICLE_HEAT
 return ugkwp::coldWallCarryCaloric(p).specificHeat(t);
#else
 return Foam::gpuThermal::coldWallApparentSpecificHeat(t,p);
#endif
}
UGKWP_COLD_CARRY_HD inline ColdWallThermalReal coldWallCarryFraction
(ColdWallThermalReal t,const Foam::gpuThermal::ColdWallSolidificationParameters& p) {
#ifdef GPU_PAIRED_PARTICLE_HEAT
 return ugkwp::coldWallCarrySolidFraction(t,p);
#else
 const GpuReal ts=Foam::gpuThermal::coldWallSolidusTemperature(p);
 const GpuReal tl=Foam::gpuThermal::coldWallLiquidusTemperature(p);
 return t<=ts?GPU_R(1.0):(t>=tl?GPU_R(0.0):(tl-t)/p.mushyRangeK);
#endif
}
#undef UGKWP_COLD_CARRY_HD
