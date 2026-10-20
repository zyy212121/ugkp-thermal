#pragma once
// One requested interval, one physical execution. Storage is never snapshotted
// or replayed. The existing stage-completion synchronization also publishes
// structured diagnostics; it does not introduce another synchronization point.
inline int releaseSharedGasTrialStorage(SharedGasTrialStorage&) { return 0; }

__global__ void commitAcceptedWallHistory(DeviceState*state){
 auto&m=state->gasBoundaryLayerModel;const int slot=blockIdx.x*blockDim.x+threadIdx.x;
 if(slot>=state->gasBoundaryLayer.count||!m.acceptedHistory||!m.candidateHistory||!m.candidateFrame||!m.acceptedFrame||!m.candidateFrame[slot].valid)return;
 constexpr int B=ugkwp::compiledGasSpecies+4;const std::size_t offset=std::size_t(slot)*m.config.nodes*B;
 for(int i=0;i<m.config.nodes*B;++i)m.acceptedHistory[offset+i]=m.candidateHistory[offset+i];
 m.acceptedFrame[slot]=m.candidateFrame[slot];
}
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
        const auto error=cudaDeviceSynchronize();
        if(error!=cudaSuccess){setLastError("complete shared gas stage",error);return 1;}
        if(s->diagnosticOwner){
            const auto observed=s->diagnosticOwner->observeCompletedWork();
            if(observed!=cudaSuccess){setLastError("read shared gas diagnostics",observed);return 1;}
        }
        std::vector<int> cells(s->nCells),faces(s->nFaces);
        if(copyToHost(cells.data(),s->gasSpecies.cellStatus,cells.size(),"read mixture cell diagnostics")
           ||copyToHost(faces.data(),s->gasSpecies.faceStatus,faces.size(),"read mixture face diagnostics"))return 1;
        for(int location=0;location<2;++location)
        {
            const auto& values=location?faces:cells;
            for(std::size_t index=0;index<values.size();++index)
            {
                // Preserve the existing non-refinement failure check while
                // producer-specific diagnostics supply the precise cause.
                if(values[index]!=int(ugkwp::GasTransportCode::InvalidStorage))continue;
                if(s->diagnosticOwner && !s->diagnosticOwner->terminal())
                    s->diagnosticOwner->recordHostFatal(ugkp::diagnostics::makeRecord(
                        ugkp::diagnostics::Code::MissingOperator,2509,23,0,
                        location?ugkp::diagnostics::EntityKind::Face:ugkp::diagnostics::EntityKind::Cell,
                        index,-1,values[index]));
                else if(!s->diagnosticOwner) {setLastErrorText("shared gas operator/storage unavailable");return 1;}
            }
        }
        if(s->diagnosticOwner)
        {
            if(s->diagnosticOwner->dispatchPending())return 1;
        }
        return 0;
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
