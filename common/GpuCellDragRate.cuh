#pragma once
template<class DragModel>
__device__ __forceinline__ GPU_OPERATOR_REAL cellDragInverseTimeCore
(
    const DeviceState& s,
    const GPU_OPERATOR_REAL rhoG,
    const GPU_OPERATOR_REAL gasViscosity,
    const GPU_OPERATOR_REAL epsG,
    const GPU_OPERATOR_REAL diameter,
    const GPU_OPERATOR_REAL speed,
    const DragModel model
)
{
#if GPU_OPERATOR_THERMAL
    const ugkwpGpuDrag::ThermalDragInput input
    {rhoG, epsG, gasViscosity, s.rhoSolid, diameter, speed};
    return model.inverseRelaxationTime(input);
#else
    const ugkwpGpuDrag::DragInput input
    {rhoG, gasViscosity, epsG, s.rhoSolid, diameter, speed, OfSmall,
     s.dragParameter0, s.dragParameter1, s.dragParameter2, s.dragParameter3};
    return DragModel::inverseResponseTime(input);
#endif
}
template<class DragModel>
__device__ __forceinline__ GPU_OPERATOR_REAL cellDragInverseTime
(const DeviceState& s, const int c, const GPU_OPERATOR_REAL rhoG, const GPU_OPERATOR_REAL epsG,
 const GPU_OPERATOR_REAL diameter, const GPU_OPERATOR_REAL speed, const DragModel model)
{
    const GPU_OPERATOR_REAL mu=ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]);
    return cellDragInverseTimeCore(s,rhoG,mu,epsG,diameter,speed,model);
}
