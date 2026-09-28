#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void trackParticlesLocalFaceWalkKernel(DeviceState* sp, const GPU_OPERATOR_TIME dt)
{
    DeviceState& s = *sp;
    const int nParticles = clampRange(*s.particleCountDevice, 0, s.particleCapacity);
    for
    (
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < nParticles;
        i += blockDim.x*gridDim.x
    )
    {
        trackOneParticleLocalFaceWalk(s, i, dt);
    }
}
