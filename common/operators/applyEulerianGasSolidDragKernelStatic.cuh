#pragma once
#include "GpuCellDragRate.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
template<class DragModel>
__global__ void applyEulerianGasSolidDragKernelStatic
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME dt,
    const DragModel dragModel
)
{
#include "GpuCellDragUpdate.inl"
}
