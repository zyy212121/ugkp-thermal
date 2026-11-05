#pragma once
// One requested interval, one physical execution. Stage checks stay on device;
// the public interval/Courant completion boundary publishes diagnostics before
// the host can advance its physical clock or consume its timestep selection.


#include "gasTransport/ResidentWallHistory.inl"
struct SharedGasTrialPolicy
{
    static int beginInterval(DeviceState* s) { return s->gasModelPoisoned?1:0; }
    static int begin(DeviceState* s) { return s->gasModelPoisoned?1:0; }
    static int applySources(DeviceState* s,double dt,bool nativeRefinement=false)
    {
        const int block=s->fixedCellBlockThreads;
        return applyGasGravitySource(s,(s->nCells+block-1)/block,block,dt,nativeRefinement);
    }
    static int validate(DeviceState* s,bool=false)
    {
        if(s->gasModelPoisoned)return 1;
        const auto error=ugkwp::enqueueGasStatusCheck(s->gasSpecies.cellStatus,s->nCells,
            s->gasSpecies.faceStatus,s->nFaces,s->diagnostics,s->gasStatusFirstUnavailable,
            2509,23,s->gasCaptureStream);
        if(error!=cudaSuccess){setLastError("check shared gas stage status",error);return 1;}
        return 0;
    }
    static int complete(DeviceState* s)
    {
        // Pair the bounded diagnostic readback with this stream's single wait.
        // Error paths also drain queued work before the public API returns.
        const auto error=s->diagnosticOwner
            ?s->diagnosticOwner->observeCompletedWork(s->gasCaptureStream)
            :cudaStreamSynchronize(s->gasCaptureStream);
        if(error!=cudaSuccess){setLastError("complete shared gas stage",error);return 1;}
        if(s->diagnosticOwner&&s->diagnosticOwner->dispatchPending())return 1;
        return s->gasModelPoisoned?1:0;
    }
    static int commit(DeviceState* s) { return s->gasModelPoisoned?1:0; }
    static int commitInterval(DeviceState* s) {
        if(s->gasModelPoisoned)return 1;
        if(s->gasBoundaryLayer.enabled&&s->gasBoundaryLayerModel.acceptedHistory){
            commitAcceptedWallHistory<<<(s->gasBoundaryLayer.count+63)/64,64,0,s->gasCaptureStream>>>(s->deviceState);
            const auto error=cudaGetLastError();if(error!=cudaSuccess){setLastError("commit accepted wall history",error);return 1;}
        }
        return 0;
    }
    static const double* stageVolumes(DeviceState* s,bool) { return s->V; }
};
