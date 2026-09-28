#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
#include "GpuReductionTaskCount.cuh"

#include "GpuReductionTaskWrite.cuh"

#include "GpuReductionTaskMaterialize.cuh"

__global__ void publishCsrReductionTaskCountKernel(DeviceState* sp)
{
    DeviceState& s = *sp;
    *s.csrHeavyTaskCount = s.csrCellTaskOffset[s.nCells];
}
