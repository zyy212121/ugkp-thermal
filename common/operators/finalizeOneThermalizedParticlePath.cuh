#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<bool StuckPath>
__device__ void finalizeOneThermalizedParticlePath
(
    DeviceState& s,
    const int i,
    const GPU_OPERATOR_REAL candidateUx,
    const GPU_OPERATOR_REAL candidateUy,
    const GPU_OPERATOR_REAL candidateUz,
    const GPU_OPERATOR_REAL poolMeanUx,
    const GPU_OPERATOR_REAL poolMeanUy,
    const GPU_OPERATOR_REAL poolMeanUz
)
{
    if (StuckPath)
    {
        finalizeOneThermalizedStuckParticle
        (
            s, i,
            candidateUx, candidateUy, candidateUz,
            poolMeanUx, poolMeanUy, poolMeanUz
        );
    }
    else
    {
        finalizeOneThermalizedMobileParticle
        (
            s, i,
            candidateUx, candidateUy, candidateUz
        );
    }
}
