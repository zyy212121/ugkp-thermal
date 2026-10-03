#pragma once
#include "GpuPoolMomentOperations.cuh"
#include "GpuCollisionProbability.cuh"
// Shared split-directory S1 protocol; scalar-tree/component reduction remains a compile-time policy.
#if GPU_POOL_STATIC_DIRECTORY
template<bool DirectParticleIndex, bool AddToExistingPool, bool HeavyReductionEnabled>
__global__ void accumulatePoissonPoolSplitSegmentByCellKernel
#else
template<bool DirectParticleIndex, bool AddToExistingPool>
__global__ void accumulatePoissonPoolSplitSegmentKernel
#endif
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

    const int start = DirectParticleIndex
      ? s.preBaseCellOffset[c] : s.cellParticleOffset[c];
    const int end = DirectParticleIndex
      ? s.preBaseCellOffset[c + 1] : s.cellParticleOffset[c + 1];
    if (start >= end) return;
#if GPU_POOL_STATIC_DIRECTORY
    if constexpr (HeavyReductionEnabled) return;
#else
    if (s.csrHeavyReductionEnabled != 0) return;
#endif
    __shared__ GPU_OPERATOR_REAL cellCollisionProbability;

    if (threadIdx.x == 0)
    {
        cellCollisionProbability = poissonCollisionProbabilityForCell(s, c, dt);
    }

    __syncthreads();

    const GPU_OPERATOR_REAL prob = cellCollisionProbability;
    if (prob <= GPU_OPERATOR_R(0.0)) return;
    GPU_OPERATOR_REAL sums[8] = {};
    for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x)
    {
        const int i = DirectParticleIndex
          ? pos
          : s.sortedParticleIndex[pos];
        accumulateOnePoolParticle<true>
        (s, c, i, prob, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6], sums[7]);
    }

    extern __shared__ GPU_OPERATOR_REAL scratch[];
    reducePoolMoments<GPU_POOL_SPLIT_REDUCTION_TOPOLOGY>(sums, scratch);
    if (threadIdx.x == 0) publishPoolCell<AddToExistingPool>(s, c, sums);
}
