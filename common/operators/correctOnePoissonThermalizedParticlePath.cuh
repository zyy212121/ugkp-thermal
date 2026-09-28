#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
#include "../GpuCollisionPoolCorrection.cuh"

__device__ void correctOnePoissonThermalizedMobileParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    correctOnePoissonThermalizedParticlePath<false>
    (
        s,
        i,
        applyThetaDrag
    );
}

__device__ void correctOnePoissonThermalizedStuckParticle
(
    DeviceState& s,
    const int i,
    const bool applyThetaDrag
)
{
    correctOnePoissonThermalizedParticlePath<true>
    (
        s,
        i,
        applyThetaDrag
    );
}

__global__ void correctPoissonThermalizedMobileParticlesKernel
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
        if (s.pStuck[i] != 0)
        {
            continue;
        }
        correctOnePoissonThermalizedMobileParticle
        (
            s,
            i,
            applyThetaDrag != 0
        );
    }
}

__global__ void correctPoissonThermalizedStuckParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
)
{
    DeviceState& s = *sp;
    const int selectedStuckCount =
        clampRange(*s.compactCountDevice, 0, s.particleCapacity);
    for
    (
        int pos = blockIdx.x*blockDim.x + threadIdx.x;
        pos < selectedStuckCount;
        pos += blockDim.x*gridDim.x
    )
    {
        const int i = s.compactPStatus[pos];
        if (i < 0 || i >= s.particleCapacity)
        {
            asm("trap;");
        }
        correctOnePoissonThermalizedStuckParticle
        (
            s,
            i,
            applyThetaDrag != 0
        );
    }
}
