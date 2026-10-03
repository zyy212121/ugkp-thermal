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

    GPU_OPERATOR_REAL mass = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momX = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momY = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL momZ = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL energy = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL diameter = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL diameter2 = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL count = GPU_OPERATOR_R(0.0);

    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = s.sortedParticleIndex[pos];

        accumulateOnePoolParticle<true>
        (s, c, i, prob, mass, momX, momY, momZ, energy, diameter, diameter2, count);

    }

    GPU_OPERATOR_REAL sums[8] = {mass, momX, momY, momZ, energy, diameter, diameter2, count};

    extern __shared__ GPU_OPERATOR_REAL scratch[];
    reducePoolMoments<PoolReductionTopology::sharedTree>(sums, scratch);
    if (threadIdx.x == 0) publishPoolCell<false>(s, c, sums);
}
