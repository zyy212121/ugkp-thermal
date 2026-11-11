#pragma once
// The gas pruned tree and thermal full tree retain their original arithmetic,
// full-warp contract, barriers and precision. Policy selection is compile-time.
template<int NumComponents, class ReductionReal>
__device__ void blockReduceComponentSums
(
    ReductionReal (&sums)[NumComponents],
    ReductionReal* warpPartials
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31)/32;
    constexpr unsigned int fullWarpMask = 0xffffffffu;

    // All threads participate; both strategies require full warps.
    if ((blockDim.x & 31) != 0)
    {
        asm("trap;");
    }
    __syncwarp(fullWarpMask);

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            const ReductionReal other =
                __shfl_down_sync(fullWarpMask, sums[component], offset);
#if GPU_BLOCK_REDUCTION_PRUNED_TREE
            if (lane < offset)
#else
            if (lane + offset < 32)
#endif
            {
                sums[component] += other;
            }
        }
    }

    if (lane == 0)
    {
        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            warpPartials[component*warpCount + warp] = sums[component];
        }
    }

    __syncthreads();

    if (warp == 0)
    {
        __syncwarp(fullWarpMask);

        #pragma unroll
        for (int component = 0; component < NumComponents; ++component)
        {
            ReductionReal value =
                lane < warpCount
              ? warpPartials[component*warpCount + lane]
              : ReductionReal(0);

#if GPU_BLOCK_REDUCTION_PRUNED_TREE
            int firstOffset = 16;
            while (firstOffset >= warpCount) firstOffset >>= 1;
#else
            const int firstOffset = 16;
#endif
            for (int offset = firstOffset; offset > 0; offset >>= 1)
            {
                const ReductionReal other =
                    __shfl_down_sync(fullWarpMask, value, offset);
#if GPU_BLOCK_REDUCTION_PRUNED_TREE
                if (lane < offset && lane + offset < warpCount)
#else
                if (lane + offset < 32)
#endif
                {
                    value += other;
                }
            }

            if (lane == 0)
            {
                sums[component] = value;
            }
        }
    }
}
