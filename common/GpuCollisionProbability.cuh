#pragma once
#include "ParticleResidentInterval.H"
__device__ __forceinline__ GPU_OPERATOR_REAL poissonCollisionProbabilityForCell
(DeviceState& s, const int c, const GPU_OPERATOR_TIME dt)
{
#ifdef UGKP_RESIDENT_CSR_CHOICE
    const GPU_OPERATOR_TIME interval=s.csrHeavyReductionMode==2&&s.csrAutomaticRuntime
        ?s.csrAutomaticRuntime->timeStep:ugkwp::particleResidentStepDt(s,dt);
#else
    const GPU_OPERATOR_TIME interval=ugkwp::particleResidentStepDt(s,dt);
#endif
    const GPU_OPERATOR_REAL tauColl = granularCollisionTauFromCellDevice(s, c);
    return (!(tauColl < GPU_OPERATOR_R(0.5)*OfGreat) || tauColl <= OfSmall)
      ? GPU_OPERATOR_R(0.0) : clampRange(GPU_OPERATOR_R(1.0) - exp(-interval/tauColl), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
}
