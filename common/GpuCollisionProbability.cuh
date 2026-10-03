#pragma once
__device__ __forceinline__ GPU_OPERATOR_REAL poissonCollisionProbabilityForCell
(DeviceState& s, const int c, const GPU_OPERATOR_TIME dt)
{
    const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);
    return (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
      ? GPU_OPERATOR_R(0.0) : clampRange(GPU_OPERATOR_R(1.0) - exp(-dt/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
}
