#pragma once
__device__ void writeCsrReductionTask(DeviceState& s, const int task,
    const int cell, const int begin, const int end, const CsrReductionTaskSource source)
{
    if (task < 0) asm("trap;");
    CsrReductionTask descriptor = {cell, begin, end, static_cast<int>(source)};
    s.csrReductionTasks[task] = descriptor;
}
