#pragma once
#include "GpuParticleFields.cuh"
#include "operators/swapParticlePointerDevice.cuh"
__host__ __device__ __forceinline__ void swapParticleBuffersDevice(DeviceState& s)
{
#define GPU_SWAP_FIELD(src, dest, unused) swapParticlePointerDevice(s.src, s.dest);
    GPU_PARTICLE_FIELDS_PRIMARY_BEFORE_CONTACT(GPU_SWAP_FIELD)
    GPU_PARTICLE_FIELDS_PRIMARY_AFTER_CONTACT(GPU_SWAP_FIELD)
    ParticleFieldExtension::swap(s);
    GPU_PARTICLE_FIELDS_IDENTITY(GPU_SWAP_FIELD)
#undef GPU_SWAP_FIELD
}
__global__ void commitCellLocalParticleBuffersKernel(DeviceState* sp)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    int compactCount = s.compactCellOffset[s.nCells];
    compactCount = compactCount < 0 ? 0 : compactCount;
    compactCount =
        compactCount > s.particleCapacity ? s.particleCapacity : compactCount;
    *s.particleCountDevice = compactCount;

    swapParticleBuffersDevice(s);
}

__global__ void commitSelectedParticleBuffersKernel(DeviceState* sp)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
    const int compactCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    *s.particleCountDevice = compactCount;

    swapParticleBuffersDevice(s);
}

