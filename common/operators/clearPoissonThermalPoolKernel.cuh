#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void clearPoissonThermalPoolKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    s.poolThermalCount[c] = 0;
    s.poolThermalSumUx[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumUy[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumUz[c] = GPU_OPERATOR_R(0.0);
    s.poolThermalSumU2[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolSampleTargetCount[c] = 0;
    s.poissonPoolMass[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomX[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomY[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolMomZ[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolEnergy[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolDiameter[c] = GPU_OPERATOR_R(0.0);
    s.poissonPoolDiameter2[c] = GPU_OPERATOR_R(0.0);
}

template<int NumComponents>
__device__ void blockReduceComponentSums
(
    GPU_OPERATOR_REAL (&sums)[NumComponents],
    GPU_OPERATOR_REAL* warpPartials
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
            const GPU_OPERATOR_REAL other =
                __shfl_down_sync(fullWarpMask, sums[component], offset);
            if (lane + offset < 32)
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
            GPU_OPERATOR_REAL value =
                lane < warpCount
              ? warpPartials[component*warpCount + lane]
              : GPU_OPERATOR_R(0.0);

            for (int offset = 16; offset > 0; offset >>= 1)
            {
                const GPU_OPERATOR_REAL other =
                    __shfl_down_sync(fullWarpMask, value, offset);
                if (lane + offset < 32)
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
