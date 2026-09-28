#include "GpuCollisionPoolTarget.cuh"
#pragma once
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

__device__ void appendSelectedStuckParticleIndex
(
    DeviceState& s,
    const int i
)
{
    if (s.pStuck[i] == 0)
    {
        return;
    }

    const int slot = atomicAdd(s.compactCountDevice, 1);
    if (slot < 0 || slot >= s.particleCapacity)
    {
        asm("trap;");
    }
    s.compactPStatus[slot] = i;
}
