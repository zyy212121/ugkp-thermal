#pragma once

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
