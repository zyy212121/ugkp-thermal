#pragma once
#include "GpuCollisionCentralReduction.cuh"
enum class PoolReductionTopology { sharedTree, warpComponents };
template<PoolReductionTopology Topology>
__device__ __forceinline__ void reducePoolMoments
(float (&sums)[8], float* scratch)
{
    blockReduceCollisionMoments<8>(sums, scratch);
}
template<bool Add>
__device__ __forceinline__ void publishPoolCell
(DeviceState& s, const int c, const float (&sums)[8])
{
    if constexpr (Add)
    {
        const auto merged=mergeCollisionCentralMoments(
            {s.poissonPoolMass[c],s.poissonPoolMomX[c],s.poissonPoolMomY[c],
             s.poissonPoolMomZ[c],s.poissonPoolEnergy[c]},
            {sums[0],sums[1],sums[2],sums[3],sums[4]});
        s.poissonPoolMass[c]=merged.mass;
        s.poissonPoolMomX[c]=merged.ux;s.poissonPoolMomY[c]=merged.uy;
        s.poissonPoolMomZ[c]=merged.uz;s.poissonPoolEnergy[c]=merged.q;
        s.poissonPoolDiameter[c] += sums[5];
        s.poissonPoolDiameter2[c] += sums[6];
        s.poolThermalCount[c] += collisionCountFromWord(sums[7]);
    }
    else
    {
        s.poissonPoolMass[c] = sums[0];
        s.poissonPoolMomX[c] = sums[1];
        s.poissonPoolMomY[c] = sums[2];
        s.poissonPoolMomZ[c] = sums[3];
        s.poissonPoolEnergy[c] = sums[4];
        s.poissonPoolDiameter[c] = sums[5];
        s.poissonPoolDiameter2[c] = sums[6];
        s.poolThermalCount[c] = collisionCountFromWord(sums[7]);
    }
}
__device__ __forceinline__ void publishPoolPartial
(DeviceState& s, const int task, const float (&sums)[8])
{
    #pragma unroll
    for (int k=0; k<8; ++k) s.csrCollisionPartials[8u*static_cast<size_t>(task)+k] = sums[k];
}
__device__ __forceinline__ void zeroPoolPartial(DeviceState& s, const int task)
{
    const float zero[8] = {};
    publishPoolPartial(s, task, zero);
}
