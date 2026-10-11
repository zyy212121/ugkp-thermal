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

#ifdef GPU_PAIRED_PARTICLE_HEAT
template<bool PairedHeat = false>
#endif
__global__ void correctPoissonThermalizedMobileParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
#ifdef GPU_PAIRED_PARTICLE_HEAT
    , const GPU_OPERATOR_TIME heatDt = 0
#endif
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
#ifdef GPU_PAIRED_PARTICLE_HEAT
        if constexpr (PairedHeat)
        {
            // Capture before correction: only the sampled-stuck subset waits
            // for its finalizer, which can detach it/change the caloric law.
            const bool deferredHeatCandidate =
                s.pStuck[i] != 0 && s.pStatus[i] == 2;
            if (s.pStuck[i] == 0)
            {
                correctOnePoissonThermalizedMobileParticle
                (
                    s, i, applyThetaDrag != 0
                );
            }
            if (!deferredHeatCandidate)
                accumulatePairedParticleHeatCandidate(s, i, heatDt);
        }
        else
#endif
        {
            // Keep the heat-disabled specialization's original body. A runtime
            // branch alone would still increase its kernel register allocation.
            if (s.pStuck[i] != 0)
            {
                continue;
            }
            correctOnePoissonThermalizedMobileParticle
            (
                s, i, applyThetaDrag != 0
            );
        }
    }
}

#ifdef GPU_PAIRED_PARTICLE_HEAT
template<bool PairedHeat = false>
#endif
__global__ void correctPoissonThermalizedStuckParticlesKernel
(
    DeviceState* sp,
    const int applyThetaDrag
#ifdef GPU_PAIRED_PARTICLE_HEAT
    , const GPU_OPERATOR_TIME heatDt = 0
#endif
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
#ifdef GPU_PAIRED_PARTICLE_HEAT
        if constexpr (PairedHeat)
            accumulatePairedParticleHeatCandidate(s, i, heatDt);
#endif
    }
}
