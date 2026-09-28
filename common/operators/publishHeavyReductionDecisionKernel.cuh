#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void publishHeavyReductionDecisionKernel
(
    DeviceState* sp,
    const int active
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        sp->csrHeavyReductionActive = active;
        sp->csrHeavyReductionEnabled = active;
    }
}
