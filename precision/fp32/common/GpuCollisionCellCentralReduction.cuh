#pragma once
#include "GpuCollisionCentralReduction.cuh"
__device__ int collisionDirectoryCount(const DeviceState& s,int c,int directory)
{
    if(!directory)return s.cellParticleOffset[c+1]-s.cellParticleOffset[c];
    const int base=s.preBaseCellOffset[c+1]-s.preBaseCellOffset[c];
    return base+(directory==2?0:s.cellParticleOffset[c+1]-s.cellParticleOffset[c]);
}
__device__ int collisionDirectoryParticle(const DeviceState& s,int c,int pos,int directory)
{
    if(!directory)return s.sortedParticleIndex[s.cellParticleOffset[c]+pos];
    const int base=s.preBaseCellOffset[c],count=s.preBaseCellOffset[c+1]-base;
    return pos<count?base+pos:s.sortedParticleIndex[s.cellParticleOffset[c]+pos-count];
}
template<bool SelectedPool>
__global__ void reduceCollisionCellCentralKernel(DeviceState* sp,int directory)
{
    DeviceState& s=*sp;const int c=blockIdx.x;
    float q[8]={};
    const int count=collisionDirectoryCount(s,c,directory);
    for(int pos=threadIdx.x;pos<count;pos+=blockDim.x)
    {
        const int i=collisionDirectoryParticle(s,c,pos,directory);
        if(i<0||i>=s.particleCapacity||s.pCellId[i]!=c
           ||(SelectedPool?s.pStatus[i]!=2:s.pStatus[i]==0))continue;
        const float m=clampMin(finiteOr(s.pm[i],0.0f),0.0f);
        const float ux=finiteOr(s.pux[i],0.0f),uy=finiteOr(s.puy[i],0.0f),uz=finiteOr(s.puz[i],0.0f);
        const float theta=GPU_POOL_PARTICLE_THETA(s,i);
        const float d=clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),1.0e-12f);
        const float one[8]={m,ux,uy,uz,1.5f*m*theta,m*d,m*d*d,collisionCountWord(1)};
        mergeCollisionMomentGroup(q,one);
    }
    extern __shared__ float scratch[];
    blockReduceCollisionMoments(q,scratch);
    if(!threadIdx.x)
    {
        if constexpr(SelectedPool)publishPoolCell<false>(s,c,q);
        else s.collisionCellMoments[c]={q[0],q[1],q[2],q[3],q[4]};
    }
}
