#pragma once
#include "GpuPoolMomentOperations.cuh"
#include "GpuCollisionProbability.cuh"
// Shared ordinary full-directory collision reduction.
#if GPU_POOL_STATIC_DIRECTORY
template<bool HeavyReductionEnabled>
#endif
__global__ void accumulatePoissonPoolParticlesByCellKernel
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt
)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x;

    if (c >= s.nCells)
    {
        return;
    }

    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    if (start >= end) return;
#if GPU_POOL_STATIC_DIRECTORY
    if constexpr (HeavyReductionEnabled)
    {
        return;
    }
#else
    if (s.csrHeavyReductionEnabled != 0)
    {
        return;
    }

#endif
    __shared__ GPU_OPERATOR_REAL cellCollisionProbability;

    if (threadIdx.x == 0)
    {
        cellCollisionProbability = poissonCollisionProbabilityForCell(s, c, dt);
    }

    __syncthreads();

    const GPU_OPERATOR_REAL prob = cellCollisionProbability;

    if (prob <= GPU_OPERATOR_R(0.0))
    {
        return;
    }

    float mass = GPU_OPERATOR_R(0.0);
    float momX = GPU_OPERATOR_R(0.0);
    float momY = GPU_OPERATOR_R(0.0);
    float momZ = GPU_OPERATOR_R(0.0);
    float energy = GPU_OPERATOR_R(0.0);
    float diameter = GPU_OPERATOR_R(0.0);
    float diameter2 = GPU_OPERATOR_R(0.0);
    float count = GPU_OPERATOR_R(0.0);

    for (unsigned int cursor = static_cast<unsigned int>(start) + threadIdx.x;
         cursor < static_cast<unsigned int>(end); cursor += blockDim.x)
    {
        const int pos = static_cast<int>(cursor);
        const int i = s.sortedParticleIndex[pos];

        accumulateOnePoolParticle<true>
        (s, c, i, prob, mass, momX, momY, momZ, energy, diameter, diameter2, count);

    }

    float sums[8] = {mass, momX, momY, momZ, energy, diameter, diameter2, count};

    extern __shared__ float energyScratch[];
    reducePoolMoments<PoolReductionTopology::sharedTree>(sums, energyScratch);
    if (threadIdx.x == 0) publishPoolCell<false>(s, c, sums);
}
