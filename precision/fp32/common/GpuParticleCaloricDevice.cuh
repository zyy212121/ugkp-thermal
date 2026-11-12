#pragma once
// CHT only. The residual has the same J/kg meaning on both legacy caloric
// branches; activation/release does not reinterpret it as temperature.
template<bool CompactParticles=false>
__device__ __forceinline__ int particleColdProfileNodeCountDevice(const DeviceState& s,int i)
{
 if((CompactParticles?s.compactPStuck[i]:s.pStuck[i])==Foam::gpuThermal::particleWallMobile)return 0;
 const int f=(CompactParticles?s.compactPStuckFaceId[i]:s.pStuckFaceId[i]);
 if(f<0||f>=s.nFaces)return 0;
 const int type=s.particleStuckCandidateMask[f];
 return type==Foam::gpuThermal::particleWallSolidifyingDeposition?8
  :(type==Foam::gpuThermal::particleWallColdWall2D?64:0);
}
template<bool CompactParticles=false>
__device__ __forceinline__ ugkwp::ParticleCaloric particleCaloricDevice(const DeviceState& s,int i)
{
 ugkwp::ParticleCaloric law;
 if(particleColdProfileNodeCountDevice<CompactParticles>(s,i)){
  const auto& p=s.coldWallSolidificationParameters;
  law.phaseChange=true;law.meltingTemperature=float(p.meltingTemperatureK);
  law.mushyRange=float(p.mushyRangeK);law.latentHeat=float(p.latentHeatJkg);
  law.solidCp=float(p.solidSpecificHeatJkgK);
 }
 return law;
}
template<bool CompactParticles=false>
__device__ __forceinline__ float particleMaterialEnthalpyBaseDevice(const DeviceState& s,int i)
{
 const int count=particleColdProfileNodeCountDevice<CompactParticles>(s,i);
 if(!count)return particleCaloricDevice<CompactParticles>(s,i).enthalpy(float((CompactParticles?s.compactPT[i]:s.pT[i])));
 const float* h=count==8?(CompactParticles?s.compactPColdNodeSpecificEnthalpy:s.pColdNodeSpecificEnthalpy):(CompactParticles?s.compactPCold2DNodeSpecificEnthalpy:s.pCold2DNodeSpecificEnthalpy);
 float mean=0.f;for(int n=0;n<count;++n)mean+=float(h[i*count+n])/float(count);
 return mean;
}
template<bool CompactParticles=false>
__device__ __forceinline__ float particleEffectiveSpecificEnthalpyDevice(const DeviceState& s,int i)
{return particleMaterialEnthalpyBaseDevice<CompactParticles>(s,i);}
template<bool CompactParticles=false>
__device__ __forceinline__ float particleEffectiveTemperatureDevice(const DeviceState& s,int i)
{
 if(!particleColdProfileNodeCountDevice<CompactParticles>(s,i))
  return CompactParticles?s.compactPT[i]:s.pT[i];
 return particleCaloricDevice<CompactParticles>(s,i).temperature(particleEffectiveSpecificEnthalpyDevice<CompactParticles>(s,i));
}
