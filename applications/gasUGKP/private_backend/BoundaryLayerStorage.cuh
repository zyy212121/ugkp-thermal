#pragma once
// Allocation ownership only; closure mathematics belongs to the shared core.
struct BoundaryLayerAllocation
{
    ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
    ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
    ugkwp::GasSstAuditState<double> gasSstAudit;
    double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
};
template<class State> void releaseBoundaryLayerStorage(State& s)
{
#define WALL_RELEASE(field) releaseSharedGasPointer(s.field)
    WALL_RELEASE(gasBoundaryLayer.faceSlot);WALL_RELEASE(gasBoundaryLayer.ownerSlot);
    WALL_RELEASE(gasBoundaryLayer.status);WALL_RELEASE(gasBoundaryLayer.exchange);
    WALL_RELEASE(gasBoundaryLayer.sst);WALL_RELEASE(gasBoundaryLayer.speciesFlux);
    WALL_RELEASE(gasBoundaryLayerModel.input);WALL_RELEASE(gasBoundaryLayerModel.output);
    WALL_RELEASE(gasBoundaryLayerModel.workspace);WALL_RELEASE(gasBoundaryLayerModel.status);
    WALL_RELEASE(gasBoundaryLayerModel.faces);WALL_RELEASE(gasBoundaryLayerModel.matchingOffsets);
    WALL_RELEASE(gasBoundaryLayerModel.matchingCells);WALL_RELEASE(gasBoundaryLayerModel.matchingWeights);
    WALL_RELEASE(gasWallQuadratureDistance);WALL_RELEASE(gasWallQuadratureWeight);
    WALL_RELEASE(gasSstAudit.transportK);WALL_RELEASE(gasSstAudit.transportOmega);
    WALL_RELEASE(gasSstAudit.sourceK);WALL_RELEASE(gasSstAudit.sourceOmega);
    WALL_RELEASE(gasSstAudit.constraintK);WALL_RELEASE(gasSstAudit.constraintOmega);
    WALL_RELEASE(gasSstAudit.initialTransportK);WALL_RELEASE(gasSstAudit.initialTransportOmega);
    WALL_RELEASE(gasSstAudit.initialSourceK);WALL_RELEASE(gasSstAudit.initialSourceOmega);
    WALL_RELEASE(gasSstAudit.initialConstraintK);WALL_RELEASE(gasSstAudit.initialConstraintOmega);
    WALL_RELEASE(gasSstAudit.volume);
#undef WALL_RELEASE
    s.gasBoundaryLayer={};s.gasBoundaryLayerModel={};s.gasSstAudit={};
}
