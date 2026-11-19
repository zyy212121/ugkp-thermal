#pragma once
#include "ParticleWallFaceInteraction.H"
template<bool CompactParticles=false>
__device__ __forceinline__ int particleColdProfileNodeCountDevice(const DeviceState& s,int i)
{
 if((CompactParticles?s.compactPStuck[i]:s.pStuck[i])==Foam::gpuThermal::particleWallMobile)return 0;
 const int f=(CompactParticles?s.compactPStuckFaceId[i]:s.pStuckFaceId[i]);
 if(f<0||f>=s.nFaces)return 0;
 const int type=particleWallFaceInteraction(s,f);
 return type==Foam::gpuThermal::particleWallSolidifyingDeposition?8
  :(type==Foam::gpuThermal::particleWallColdWall2D?64:0);
}
