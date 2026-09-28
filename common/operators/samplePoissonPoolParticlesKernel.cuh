#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void samplePoissonPoolParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
)
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
        sampleOnePoissonPoolParticle(s, i, applyThetaDrag != 0);
    }
}
