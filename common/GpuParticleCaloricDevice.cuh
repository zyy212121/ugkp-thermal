#pragma once


#include "wall/ParticleColdProfileNodes.cuh"
template<bool CompactParticles=false>
__device__ __forceinline__ ugkwp::ParticleCaloric particleCaloricDevice(const DeviceState& s,int i)
{
 ugkwp::ParticleCaloric law;
 if(particleColdProfileNodeCountDevice<CompactParticles>(s,i)){
  const auto& p=s.coldWallSolidificationParameters;
  law.phaseChange=true;law.meltingTemperature=double(p.meltingTemperatureK);
  law.mushyRange=double(p.mushyRangeK);law.latentHeat=double(p.latentHeatJkg);
  law.solidCp=double(p.solidSpecificHeatJkgK);
 }
 return law;
}
template<bool CompactParticles=false>
__device__ __forceinline__ double particleMaterialEnthalpyBaseDevice(const DeviceState& s,int i)
{
 const int count=particleColdProfileNodeCountDevice<CompactParticles>(s,i);
 if(!count)return particleCaloricDevice<CompactParticles>(s,i).enthalpy(double((CompactParticles?s.compactPT[i]:s.pT[i])));
 double mean=0.;
 if(count==8){
  const double* h=s.pColdNodeSpecificEnthalpy+Foam::gpuWall::coldWallSlot(s,i,CompactParticles)*8;
  for(int n=0;n<8;++n)mean+=h[n]/8.;
 }else{
  const float* h=(CompactParticles?s.compactPCold2DNodeSpecificEnthalpy:s.pCold2DNodeSpecificEnthalpy)+i*count;
  for(int n=0;n<count;++n)mean+=double(h[n])/double(count);
 }
 return mean;
}
template<bool CompactParticles=false>
__device__ __forceinline__ double particleEffectiveSpecificEnthalpyDevice(const DeviceState& s,int i)
{return particleMaterialEnthalpyBaseDevice<CompactParticles>(s,i)+(particleColdProfileNodeCountDevice<CompactParticles>(s,i)==8?0.:(CompactParticles?s.compactPThermalEnthalpyResidual[i]:s.pThermalEnthalpyResidual[i]));}
template<bool CompactParticles=false>
__device__ __forceinline__ double particleEffectiveTemperatureDevice(const DeviceState& s,int i)
{return particleCaloricDevice<CompactParticles>(s,i).temperature(particleEffectiveSpecificEnthalpyDevice<CompactParticles>(s,i));}
