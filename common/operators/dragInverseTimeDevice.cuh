#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
template<class DragModel>
__device__ GPU_OPERATOR_REAL dragInverseTimeDevice
(
    const DeviceState& s,
    const int c,
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
        ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]),
        s.rhoSolid,
        diameter,
        relativeSpeed
    };
    return model.inverseRelaxationTime(input);
}
