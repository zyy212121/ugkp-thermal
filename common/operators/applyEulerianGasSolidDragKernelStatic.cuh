#pragma once
#include "gasTransport/ParticleHeatReservoir.H"
#include "ParticleResidentInterval.H"
#include "GpuCellDragRate.cuh"

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
