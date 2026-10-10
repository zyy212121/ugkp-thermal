#pragma once
#include <cstddef>
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

// Configure-only publication. Uploading initial fields deliberately scrubs the
// host calculation scalars, so copying the complete DeviceState here would
// destroy valid device physics/numerics. Publish only this extension's fields.
// No trial may exist yet. The enabled wall view is copied last; on any copy
// failure the caller poisons the resident and retains ownership until destroy.
inline int syncBoundaryLayerConfiguration(DeviceState* state)
{
    if(!state || !state->deviceState)
    {setLastErrorText("cannot publish boundaryLayer to a null device state; destroy resident");return 1;}
#define WALL_PUBLISH(field) \
    do { \
        const cudaError_t error=cudaMemcpy( \
            reinterpret_cast<unsigned char*>(state->deviceState)+offsetof(DeviceState,field), \
            &state->field,sizeof(state->field),cudaMemcpyHostToDevice); \
        if(error!=cudaSuccess) { \
            setLastError("publish boundaryLayer " #field "; destroy resident",error); \
            return 1; \
        } \
    } while(false)
    WALL_PUBLISH(gasBoundaryLayerModel);
    WALL_PUBLISH(gasSstAudit);
    WALL_PUBLISH(gasWallQuadratureDistance);
    WALL_PUBLISH(gasWallQuadratureWeight);
    WALL_PUBLISH(sstWallTreatment);
    WALL_PUBLISH(gasBoundaryLayer);
#undef WALL_PUBLISH
    return 0;
}
