#pragma once
#include "GpuGravityUpdate.cuh"
#ifndef GPU_OPERATOR_GRAVITY_DENSITY_FLOOR
#define GPU_OPERATOR_GRAVITY_DENSITY_FLOOR false
#endif
__global__ void applyGasGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<GPU_OPERATOR_GRAVITY_DENSITY_FLOOR>(*sp,c,dt);
}


__global__ void applyGasOnlyGravityKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<sp->nCells) applyGasGravityCell<GPU_OPERATOR_GRAVITY_DENSITY_FLOOR,true>(*sp,c,dt);
}
