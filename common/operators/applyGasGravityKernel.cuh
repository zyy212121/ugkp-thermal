#pragma once
#include "GpuGravityUpdate.cuh"
__global__ void applyGasGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<true>(*sp,c,dt);
}
