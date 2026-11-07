#pragma once
#include "GpuParticleFields.cuh"
#include "GpuParticleThermalCarryStorage.cuh"
#include "operators/swapParticlePointerDevice.cuh"
__host__ __device__ __forceinline__ void swapParticleBuffersDevice(DeviceState& s)
{
#define GPU_SWAP_FIELD(src, dest, unused) swapParticlePointerDevice(s.src, s.dest);
    GPU_PARTICLE_FIELDS_PRIMARY_BEFORE_CONTACT(GPU_SWAP_FIELD)
    swapParticleThermalEnthalpyResidual(s, 0);
    GPU_PARTICLE_FIELDS_PRIMARY_AFTER_CONTACT(GPU_SWAP_FIELD)
    ParticleFieldExtension::swap(s);
    GPU_PARTICLE_FIELDS_IDENTITY(GPU_SWAP_FIELD)
#undef GPU_SWAP_FIELD
}
// Both the host mirror and downstream device consumers need the directory
// identity after compaction. Device-controlled multi-step callers must not
// infer it from the number of API calls made on the host.
template<class State>
__host__ __device__ auto publishParticleBaseDirectory(State& s,int ready,int)
    ->decltype(s.preBaseDirectoryReady=ready,void()){s.preBaseDirectoryReady=ready;}
template<class State>
__host__ __device__ void publishParticleBaseDirectory(State&,int,long){}
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
    publishParticleBaseDirectory(s,1,0);
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
    publishParticleBaseDirectory(s,0,0);
}

