
#pragma once
#include "GpuPrecisionTypes.H"
#define GPU_PIPELINE_REAL GpuEnergy
#include "GpuSegmentedMomentWorkers.cuh"


int launchCsrSegmentedMomentReduction(DeviceState* s, const int block,
    const bool gatherSurvivors = false)
{
    return launchCommonSegmentedMomentReduction(s, block,
        {gatherSurvivors ? MomentPayload::gatherSurvivors : MomentPayload::momentsOnly,
         MomentRecovery::completeHere});
}
