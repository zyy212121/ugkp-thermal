#pragma once
#include "GpuGravityUpdate.cuh"
__global__ void applyGasGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<true>(*sp,c,dt);
}

// Gas-only CHT route: particle-coupled legacy sanitization stays in the kernel above.
__global__ void applyGasOnlyGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<true,true>(*sp,c,dt);
}
