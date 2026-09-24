#pragma once
#include "CsrPersistentQueue.cuh"
// Full-descriptor directory adapter; queue protocol is shared with gas.
template<int BlockThreads, bool IndexOnly>
struct ThermalGatherOperation
{
    __device__ bool prepare(DeviceState&, int) { return true; }
    __device__ void execute(DeviceState& s, const int task)
    {
    const CsrReductionTask descriptor = s.csrReductionTasks[task];
    const int c = descriptor.cell;
    const int start = s.cellParticleOffset[c];
    const int end = s.cellParticleOffset[c + 1];
    const bool allKept = s.cellParticleCount[c] == end - start;
    if (allKept)
    {
        gatherCellLocalRangeImpl<BlockThreads, IndexOnly>(s, c, descriptor.begin, descriptor.end,
            s.compactCellOffset[c] + descriptor.begin - start, true);
    }
    else if (descriptor.begin == start)
    {
        // Exactly one block handles a filtered cell, preserving CSR order and
        // shifting every later cell using the already scanned compact offsets.
        gatherCellLocalRangeImpl<BlockThreads, IndexOnly>(s, c, start, end,
            s.compactCellOffset[c], false);
    }

    }
};

template<int BlockThreads, bool IndexOnly>
__global__ void gatherThermalSegmentedParticlesKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    ThermalGatherOperation<BlockThreads, IndexOnly> operation;
    runCsrPersistentQueue(s, *s.csrHeavyTaskCount, operation);
}
