#pragma once
template<class DragModel>
__device__ __forceinline__ GPU_OPERATOR_REAL cellDragInverseTime
(const DeviceState& s, const int c, const GPU_OPERATOR_REAL rhoG, const GPU_OPERATOR_REAL epsG,
 const GPU_OPERATOR_REAL diameter, const GPU_OPERATOR_REAL speed, const DragModel model)
{
#if GPU_OPERATOR_THERMAL
    return dragInverseTimeDevice(s, c, rhoG, epsG, diameter, speed, model);
#else
    const ugkwpGpuDrag::DragInput input =
    {rhoG, ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]), epsG, s.rhoSolid, diameter, speed, OfSmall,
     s.dragParameter0, s.dragParameter1, s.dragParameter2, s.dragParameter3};
    return DragModel::inverseResponseTime(input);
#endif
}
