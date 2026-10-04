#pragma once
#include "GpuCollisionPoolTarget.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void preparePoissonPoolSamplingKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= s.nCells)
    {
        return;
    }

    preparePoissonPoolSamplingCell(s, c);
}
