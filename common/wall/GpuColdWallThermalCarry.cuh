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

UGKWP_COLD_CARRY_HD inline double coldWallUniformCoolingBound(const double* nodes,int count) {
 double minimum=double(nodes[0]);
 for(int j=1;j<count;++j)if(double(nodes[j])<minimum)minimum=double(nodes[j]);
 return minimum>0.?-minimum:0.;
}
UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication publishColdWallUniformSpecificHeat
(double* nodes,int count,double specificHeat) {
 if(!nodes||count<=0||!(::fabs(specificHeat)<=DBL_MAX))return {false,0.};
 double accepted=0.;
 for(int j=0;j<count;++j){
  const double target=double(nodes[j])+specificHeat;
  const double stored=target;
  if(!(::fabs(double(nodes[j]))<=DBL_MAX)||!(::fabs(target)<=DBL_MAX)||!(::fabs(double(stored))<=DBL_MAX))return {false,0.};
  accepted+=(double(stored)-double(nodes[j]))/double(count);
 }
 for(int j=0;j<count;++j)nodes[j]=double(nodes[j])+specificHeat;
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
