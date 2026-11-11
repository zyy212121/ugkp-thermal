#pragma once
#include "GpuCollisionPoolParticle.cuh"
// Compatibility name; all particle accumulation lives in one shared operation.
template<bool PoissonMode>
__device__ __forceinline__ void accumulateCsrSplitLogicalPoolParticle
(
    DeviceState& s,
    const int c,
    const int i,
    const GPU_OPERATOR_REAL collisionProbability,
    float& mass,
    float& momX,
    float& momY,
    float& momZ,
    float& energy,
    float& diameter,
    float& diameter2,
    float& count
)
{
    accumulateOnePoolParticle<PoissonMode>(s, c, i, collisionProbability,
        mass, momX, momY, momZ, energy, diameter, diameter2, count);
}
