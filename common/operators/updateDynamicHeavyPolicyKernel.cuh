#pragma once
#include "GpuHardwareReductionTile.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
__global__ void updateDynamicHeavyPolicyKernel
(
    DeviceState* sp,
    const int splitPreDirectoryActive,
    const bool scheduledInspection=false
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    DeviceState& s = *sp;
#ifdef UGKP_RESIDENT_CSR_CHOICE
    if(scheduledInspection&&!s.csrAutomaticRuntime->inspect)return;
#endif
    long long population = s.cellParticleOffset[s.nCells];
    if (splitPreDirectoryActive != 0)
    {
        population += *s.preBaseParticleCountDevice;
    }
    const int threshold = hardwareReductionTile(population, s.reductionBlockThreads,
        s.multiprocessorCount, s.lightResidentBlocksPerSm);
    if (threshold == 0) asm("trap;");
    s.dynamicHeavyThreshold = static_cast<int>(threshold);
    s.csrHeavyTileParticles = static_cast<int>(threshold);
}
