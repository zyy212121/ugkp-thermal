#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void clearSplitPreInjectionBinsKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c <= s.nCells; c += stride)
    {
        s.cellParticleCount[c] = 0;
        s.cellParticleOffset[c] = 0;
        if (c < s.nCells)
        {
            s.cellParticleWrite[c] = 0;
        }
    }
}



__global__ void initialiseSplitPreInjectionWritesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int stride = blockDim.x*gridDim.x;
    for (int c = blockIdx.x*blockDim.x + threadIdx.x; c < s.nCells; c += stride)
    {
        s.cellParticleWrite[c] = s.cellParticleOffset[c];
    }
}



#include "GpuInjectionDirectory.cuh"
