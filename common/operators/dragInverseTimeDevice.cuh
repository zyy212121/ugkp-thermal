#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class DragModel>
__device__ GPU_OPERATOR_REAL dragInverseTimeDevice
(
    const DeviceState& s,
    const GPU_OPERATOR_REAL gasDensity,
    const GPU_OPERATOR_REAL gasVolumeFraction,
    const GPU_OPERATOR_REAL diameter,
    const GPU_OPERATOR_REAL relativeSpeed,
    const DragModel& model
)
{
    const ugkwpGpuDrag::DragInput input
    {
        gasDensity,
        gasVolumeFraction,
        s.gasMu,
        s.rhoSolid,
        diameter,
        relativeSpeed
    };
    return model.inverseRelaxationTime(input);
}
