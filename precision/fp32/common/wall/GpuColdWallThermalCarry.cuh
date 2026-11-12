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
struct ColdWallHeatPublication { bool valid; float acceptedSpecificHeat; };
UGKWP_COLD_CARRY_HD inline float coldWallUniformCoolingBound(const float* nodes,int count) {
 float minimum=nodes[0];
 for(int j=1;j<count;++j)if(nodes[j]<minimum)minimum=nodes[j];
 return minimum>0.0f?-minimum:0.0f;
}
UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication publishColdWallUniformSpecificHeat
(float* nodes,int count,float specificHeat) {
 if(!nodes||count<=0||!(std::fabs(specificHeat)<=FLT_MAX))return {false,0.0f};
 float accepted=0.0f;
 for(int j=0;j<count;++j){
  const float target=nodes[j]+specificHeat;
  if(!(std::fabs(nodes[j])<=FLT_MAX)||!(std::fabs(target)<=FLT_MAX))return {false,0.0f};
  accepted+=(target-nodes[j])/float(count);
 }
 for(int j=0;j<count;++j)nodes[j]+=specificHeat;
 return {true,accepted};
}
template<class Parameters>
UGKWP_COLD_CARRY_HD inline ParticleCaloric coldWallCarryCaloric(const Parameters& p) {
 ParticleCaloric c;c.phaseChange=true;c.meltingTemperature=p.meltingTemperatureK;
 c.mushyRange=p.mushyRangeK;c.latentHeat=p.latentHeatJkg;c.solidCp=p.solidSpecificHeatJkgK;return c;
}
template<class Parameters>
UGKWP_COLD_CARRY_HD inline float coldWallCarrySolidFraction(float temperature,const Parameters& p) {
 const float tl=float(p.meltingTemperatureK)+0.5f*float(p.mushyRangeK);
 const float f=(tl-temperature)/float(p.mushyRangeK);return f<0.0f?0.0f:(f>1.0f?1.0f:f);
}
}
using ColdWallThermalReal=GpuReal;
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
 return ugkwp::coldWallCarrySolidFraction(t,p);
}
#undef UGKWP_COLD_CARRY_HD
