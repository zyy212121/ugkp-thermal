// Immutable fa71f477 reference for old/new regression, never production.
template<int NumComponents>
__device__ void blockReduceComponentSums
(
    double (&sums)[NumComponents],
    double* warpPartials
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpCount = (blockDim.x + 31)/32;
    constexpr unsigned int fullWarpMask = 0xffffffffu;





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
            const double other =
                __shfl_down_sync(fullWarpMask, sums[component], offset);
            if (lane < offset)
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
            double value =
                lane < warpCount
              ? warpPartials[component*warpCount + lane]
              : 0.0;

            int firstOffset = 16;
            while (firstOffset >= warpCount) firstOffset >>= 1;
            for (int offset = firstOffset; offset > 0; offset >>= 1)
            {
                const double other =
                    __shfl_down_sync(fullWarpMask, value, offset);
                if (lane < offset && lane + offset < warpCount)
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
