#pragma once
#include <cstddef>
// Allocation ownership only; closure mathematics belongs to the shared core.
template<class State> int releaseBoundaryLayerStorage(State& s)
{
    int result=0;
#define WALL_RELEASE(field) result|=releaseSharedGasPointer(s.field)
    WALL_RELEASE(gasBoundaryLayer.faceSlot);WALL_RELEASE(gasBoundaryLayer.ownerSlot);
    WALL_RELEASE(gasBoundaryLayer.status);WALL_RELEASE(gasBoundaryLayer.exchange);
    WALL_RELEASE(gasBoundaryLayer.sst);WALL_RELEASE(gasBoundaryLayer.speciesFlux);
    WALL_RELEASE(gasBoundaryLayerModel.input);WALL_RELEASE(gasBoundaryLayerModel.output);
    WALL_RELEASE(gasBoundaryLayerModel.profile);
    WALL_RELEASE(gasBoundaryLayerModel.workspace);WALL_RELEASE(gasBoundaryLayerModel.status);
    WALL_RELEASE(gasBoundaryLayerModel.faces);WALL_RELEASE(gasBoundaryLayerModel.matchingOffsets);
    WALL_RELEASE(gasBoundaryLayerModel.matchingCells);WALL_RELEASE(gasBoundaryLayerModel.matchingWeights);
    WALL_RELEASE(gasWallQuadratureDistance);WALL_RELEASE(gasWallQuadratureWeight);
#undef WALL_RELEASE
    if(!result){s.gasBoundaryLayer={};s.gasBoundaryLayerModel={};}
    return result;
}




inline int releaseBoundaryLayerStorage(DeviceState& s)
{
    const int resident=releaseBoundaryLayerStorage<DeviceState>(s);
    if(!resident)
    {
        s.gasBoundaryLayerLaunch=&launchBoundaryLayer<false>;
        s.gasBoundaryLayerSnapshot=&sharedGasTrialDetail::snapshotWallRecords<false,DeviceState>;
    }
    return releaseBoundaryLayerStorage(s.gasTrial.rejectedWall)|resident;
}
inline void discardBoundaryLayerCandidate(DeviceState* s,BoundaryLayerAllocation& candidate)
{
    if(releaseBoundaryLayerStorage(candidate))
    {
        s->gasTrial.rejectedWall=candidate;candidate={};s->gasModelPoisoned=true;
    }
}

// Configure-only publication. Uploading initial fields deliberately scrubs the
// host calculation scalars, so copying the complete DeviceState here would
// destroy valid device physics/numerics. Publish only this extension's fields.


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
    WALL_PUBLISH(gasWallQuadratureDistance);
    WALL_PUBLISH(gasWallQuadratureWeight);
    WALL_PUBLISH(sstWallTreatment);
    WALL_PUBLISH(gasBoundaryLayer);
#undef WALL_PUBLISH
    return 0;
}
