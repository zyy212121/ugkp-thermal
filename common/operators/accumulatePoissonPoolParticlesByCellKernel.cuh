#include "GpuCollisionPoolParticle.cuh"
#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
#include "../GpuCollisionPoolCell.cuh"

#include "../GpuCollisionPoolSplitCell.cuh"

#include "../GpuCollisionPoolAtomic.cuh"

#include "GpuCollisionPoolRange.cuh"

template<bool PoissonMode>
__device__ __forceinline__ void accumulateCsrSplitLogicalPoolParticle
(
    DeviceState& s,
    const int c,
    const int i,
    const GPU_OPERATOR_REAL collisionProbability,
    GPU_OPERATOR_REAL& mass,
    GPU_OPERATOR_REAL& momX,
    GPU_OPERATOR_REAL& momY,
    GPU_OPERATOR_REAL& momZ,
    GPU_OPERATOR_REAL& energy,
    GPU_OPERATOR_REAL& diameter,
    GPU_OPERATOR_REAL& diameter2,
    GPU_OPERATOR_REAL& count
)
{
    accumulateOnePoolParticle<PoissonMode>(s, c, i, collisionProbability,
        mass, momX, momY, momZ, energy, diameter, diameter2, count);
}
