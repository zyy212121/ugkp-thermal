#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ bool mobilePackingParticleEligible
(
    const DeviceState& s,
    const int i
)
{
    return s.pStatus[i] == 1
#if GPU_OPERATOR_THERMAL
        && s.pStuck[i] == 0
#endif
        ;
}
