// Compatibility path for existing thermal includes. Authoritative implementation: common/GpuSegmentedMomentWorkers.cuh.
#pragma once
#include "GpuPrecisionTypes.H"
#define GPU_PIPELINE_REAL GpuReal
#include "GpuSegmentedMomentWorkers.cuh"

// Legacy entry point remains an adapter; all work belongs to the common backend.
int launchCsrSegmentedMomentReduction(DeviceState* s, const int block,
    const bool gatherSurvivors = false)
{
    return launchCommonSegmentedMomentReduction(s, block,
        {gatherSurvivors ? MomentPayload::gatherSurvivors : MomentPayload::momentsOnly,
         MomentRecovery::completeHere});
}
