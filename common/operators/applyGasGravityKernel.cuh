#pragma once
#include "GpuGravityUpdate.cuh"
__global__ void applyGasGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<true>(*sp,c,dt);
}

// Both resident and gas-only routes update gas, including zero-particle thermal coupling.
__global__ void applyGasOnlyGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<true,true>(*sp,c,dt);
}
