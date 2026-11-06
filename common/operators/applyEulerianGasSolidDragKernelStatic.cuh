#pragma once
#include "ParticleResidentInterval.H"
#include "GpuCellDragRate.cuh"
// One operator implementation; scalar/time adapters are compile-time only.
template<class DragModel>
__global__ void applyEulerianGasSolidDragKernelStatic
(
    DeviceState* sp,
    const GPU_OPERATOR_TIME requestedDt,
    const DragModel dragModel
)
{
    const auto dt=ugkwp::particleResidentStepDt(*sp,requestedDt);
#include "GpuCellDragUpdate.inl"
}
