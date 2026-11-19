#pragma once
#include "gpu/AdvanceState.cuh"
namespace {
using ugkp::gpu::AdvanceState;
using ugkp::gpu::AdvanceRequest;
using ugkp::gpu::AdvanceStop;
using ugkp::gpu::SteadyAdvanceControls;

__global__ void selectSteadyWindowStart(AdvanceState* state,SteadyAdvanceControls controls,
    const ugkp::diagnostics::DeviceContext* diagnostics,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;
    cudaGraphSetConditional(handle,!ugkp::diagnostics::hasFatal(diagnostics)
        &&(!state->windowSamples||state->windowSamples==controls.windowSamples));
}
__global__ void resetSteadyWindowClock(AdvanceState* state){
    if(!threadIdx.x){state->windowSamples=0;state->windowDuration=0;state->windowReady=false;}
}
template<class State>
__global__ void acceptSteadyStep(State* physics,AdvanceState* state,const AdvanceRequest* request,
    ugkp::gpu::AdvanceProgressPoint* progress,SteadyAdvanceControls controls,bool sampling,
    cudaGraphConditionalHandle check,cudaGraphConditionalHandle loop){
    if(threadIdx.x)return;
    const auto* diagnostics=physics->diagnostics;
    if(sampling&&!ugkp::diagnostics::hasFatal(diagnostics)){
        ++state->windowSamples;
        state->windowDuration=__dadd_rn(state->windowDuration,state->dt);
    }
    ugkp::gpu::commitPhysicalStep(physics,state,request,progress);
    const bool end=state->time>=request->endTime-0.5*state->clockDt;
    const bool due=sampling?state->windowSamples==controls.windowSamples
        :(state->steps-state->initialSteps)%controls.residualInterval==0;
    const bool measure=!ugkp::diagnostics::hasFatal(diagnostics)&&(due||end);
    cudaGraphSetConditional(check,measure);
    if(!measure)cudaGraphSetConditional(loop,ugkp::gpu::advanceMayContinue(*state,*request,diagnostics));
}
template<class Real>
__global__ void decideSteadyResidual(AdvanceState* state,const Real* residual,const double* scales,
    SteadyAdvanceControls controls,int components){
    if(threadIdx.x)return;
    bool converged=state->steps-state->initialSteps>=controls.minimumIterations;
    Real maximumRelative=0;
    for(int k=0;k<components;++k){
        const Real value=residual[k],scale=Real(scales[k]);


        Real limit;
        if constexpr(sizeof(Real)==sizeof(float))
            limit=__fadd_rn(Real(controls.absoluteTolerance),__fmul_rn(Real(controls.relativeTolerance),scale));
        else limit=__dadd_rn(Real(controls.absoluteTolerance),__dmul_rn(Real(controls.relativeTolerance),scale));
        converged=converged&&isfinite(value)&&value<=limit;
        if(scale>0)maximumRelative=fmax(maximumRelative,value/scale);
    }
    state->maximumRelative=maximumRelative;state->residualReady=true;
    if(converged)state->stop=AdvanceStop::converged;
}
template<class Real>
__global__ void decideSteadyWindow(AdvanceState* state,const Real* values,SteadyAdvanceControls controls){
    if(threadIdx.x)return;
    constexpr int fields=GpuSteadySamplingReport::fields;
    ++state->windows;
    bool stationary=state->windows>=controls.minimumWindows&&state->windowSamples==controls.windowSamples;
    Real maxChange=0,maxDrift=0;
    for(int k=0;k<fields;++k){
        const Real change=values[k],scale=values[4*fields+k];


        const Real difference=values[3*fields+k]-values[2*fields+k];
        const Real floor=sizeof(Real)==4?std::numeric_limits<Real>::min():Real(1e-300);
        const Real drift=fabs(difference)/fmax(scale,floor);
        stationary=stationary&&isfinite(change)&&isfinite(drift)
            &&change<=Real(controls.meanTolerance)&&drift<=Real(controls.inventoryTolerance);
        maxChange=fmax(maxChange,change);maxDrift=fmax(maxDrift,drift);
    }
    state->stableWindows=stationary?state->stableWindows+1:0;state->stationary=stationary;
    state->maximumMeanChange=maxChange;state->maximumInventoryDrift=maxDrift;state->windowReady=true;
    if(state->stableWindows>=controls.stableWindows)state->stop=AdvanceStop::converged;
}

cudaGraphNode_t appendSteadyAdvancePreparation(DeviceState* state,ugkp::gpu::AdvanceGraphStorage& owner,
    cudaGraph_t graph,cudaGraphNode_t dependency){
    using namespace ugkp::gpu::executionGraphDetail;
    if(!owner.steadyConfigured)throw std::logic_error("steady advance controls are not configured");
    const auto controls=owner.steadyControls;
    if(state->gasSteadyMode){
        return appendRecorded(graph,{dependency},owner.stream,[&]{
            if(ugkwpGpuResidentStrictPrepareSteadyStep(state,controls.courant,controls.maximumDt))
                throw std::runtime_error(lastError);
        });
    }
    cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,owner.graph,0,cudaGraphCondAssignDefault));
    const auto select=appendRecorded(graph,{dependency},owner.stream,[&]{
        selectSteadyWindowStart<<<1,1,0,owner.stream>>>(owner.state,controls,state->diagnostics,handle);
    });
    cudaGraph_t body=nullptr;const auto branch=addIf(graph,handle,{select},body);
    appendRecorded(body,{},owner.stream,[&]{
        const int block=state->fixedCellBlockThreads;
        beginSteadySampleWindowKernel<<<(state->nCells+block-1)/block,block,0,owner.stream>>>(state->deviceState);
        checked(cudaMemsetAsync(state->steadyBoundaryBudget,0,
            GpuSteadySamplingReport::budgetFields*sizeof(*state->steadyBoundaryBudget),owner.stream));
        resetSteadyWindowClock<<<1,1,0,owner.stream>>>(owner.state);
    });
    return branch;
}
void appendSteadyAcceptance(DeviceState* state,ugkp::gpu::AdvanceGraphStorage& owner,
    cudaGraph_t graph,cudaGraphNode_t dependency,cudaGraphConditionalHandle loop){
    using namespace ugkp::gpu::executionGraphDetail;
    const auto controls=owner.steadyControls;
    cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,owner.graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,{dependency},owner.stream,[&]{
        if(!state->gasSteadyMode){
            const int block=state->fixedCellBlockThreads;
            accumulateSteadySampleWindowKernel<<<(state->nCells+block-1)/block,block,0,owner.stream>>>(state->deviceState,1.0);
        }
        acceptSteadyStep<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,owner.request,owner.deviceProgress,
            controls,!state->gasSteadyMode,handle,loop);
    });
    cudaGraph_t checkGraph=nullptr;addIf(graph,handle,{selected},checkGraph);
    appendRecorded(checkGraph,{},owner.stream,[&]{
        const int block=state->fixedCellBlockThreads,grid=(state->nCells+block-1)/block;
        if(state->gasSteadyMode){
            ugkwp::invalidateGasBoundaryLayerPreparation(*state);
            const int faceBlock=state->fixedFaceBlockThreads;
            const int boundaryGrid=(state->nFaces-state->nInternalFaces+faceBlock-1)/faceBlock;
            if(boundaryGrid)updateLegacyGasBoundaryMirrorKernel<<<boundaryGrid,faceBlock,0,owner.stream>>>(state->deviceState,state->gasSteadyBoundaryTime);
            if(advanceGasEulerSubstage(state,0,0,state->gasSteadyBoundaryTime))throw std::runtime_error(lastError);
            steadyGasSpatialResidualKernel<<<grid,block,0,owner.stream>>>(state->deviceState);
            constexpr int components=7+ugkwp::compiledGasSpecies;
            checked(ugkwp::batchedComponentMaximum(state->gasSteadyResidual,state->nCells,components,state->gasSteadyMaximumWorkspace,owner.stream));
            decideSteadyResidual<<<1,1,0,owner.stream>>>(owner.state,
                state->gasSteadyMaximumWorkspace+size_t(ugkwp::componentMaximumParts(state->nCells))*components,
                owner.residualScales,controls,components);
        }else{
            finishSteadySampleWindowKernel<<<grid,block,0,owner.stream>>>(state->deviceState,0.0,&owner.state->windowDuration);
            constexpr int fields=GpuSteadySamplingReport::fields,components=5*fields;
            checked(ugkwp::batchedComponentStatistics(state->steadySampleScratch,state->nCells,
                components,2*fields,state->steadySampleReductionWorkspace,owner.stream));
            auto* values=state->steadySampleReductionWorkspace+size_t(ugkwp::componentMaximumParts(state->nCells))*components;
            checked(cudaMemcpyAsync(values+components,state->steadyBoundaryBudget,
                GpuSteadySamplingReport::budgetFields*sizeof(*values),cudaMemcpyDeviceToDevice,owner.stream));
            decideSteadyWindow<<<1,1,0,owner.stream>>>(owner.state,values,controls);
        }
        ugkp::gpu::finishAdvanceSubmissionStep<<<1,1,0,owner.stream>>>(owner.state,owner.request,state->diagnostics,loop);
    });
}
}

extern "C" int ugkwpGpuResidentStrictConfigureSteadyAdvance(void* handle,
    const ugkp::gpu::SteadyAdvanceControls* controls,const double* scales,int components){
    auto* state=asState(handle);
    try{
        using namespace ugkp::gpu;using executionGraphDetail::checked;
        if(!state||!state->advanceGraph||!controls)throw std::logic_error("configure the advance owner before steady controls");
        auto& owner=*state->advanceGraph;
        if(owner.pending)throw std::logic_error("cannot reconfigure active steady advance");
        if(!controls->residualInterval||!controls->windowSamples||!controls->stableWindows)
            throw std::invalid_argument("steady intervals must be positive");
        owner.clearGraph();owner.steadyControls=*controls;
        if(state->gasSteadyMode){
            if(!scales||components!=7+ugkwp::compiledGasSpecies)throw std::invalid_argument("invalid steady residual normalization");
            if(!owner.residualScales)checked(cudaMalloc(reinterpret_cast<void**>(&owner.residualScales),components*sizeof(double)));
            PinnedTransferQueue::Operation upload(owner.transfers);owner.transfers.upload(owner.residualScales,scales,components*sizeof(double));
        }
        owner.steadyConfigured=true;return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictCollectSteadyResidual(void* handle,double* residual){
    auto* state=asState(handle);
    try{
        if(!state||!state->advanceGraph||state->advanceGraph->pending)throw std::logic_error("collect advance before its residual report");
        exportSteadyValidationSnapshot(state);
        auto& owner=*state->advanceGraph;constexpr int components=7+ugkwp::compiledGasSpecies;
        ugkp::gpu::PinnedTransferQueue::Operation read(owner.transfers);
        owner.transfers.downloadConverted(residual,state->gasSteadyMaximumWorkspace
            +size_t(ugkwp::componentMaximumParts(state->nCells))*components,components);
        owner.transfers.finish();return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictCollectSteadyWindow(void* handle,GpuSteadySamplingReport* report){
    auto* state=asState(handle);
    try{
        if(!state||!state->advanceGraph||state->advanceGraph->pending)throw std::logic_error("collect advance before its window report");
        auto& owner=*state->advanceGraph;
        constexpr int fields=GpuSteadySamplingReport::fields,components=5*fields;
        using Real=std::remove_pointer_t<decltype(state->steadySampleReductionWorkspace)>;
        Real values[components+GpuSteadySamplingReport::budgetFields];
        ugkp::gpu::PinnedTransferQueue::Operation read(owner.transfers);
        owner.transfers.download(values,state->steadySampleReductionWorkspace
            +size_t(ugkwp::componentMaximumParts(state->nCells))*components,sizeof(values));
        owner.transfers.finish();
        for(int k=0;k<fields;++k){
            report->meanChange[k]=values[k];report->fluctuation[k]=values[fields+k];
            report->inventoryBegin[k]=values[2*fields+k];report->inventoryEnd[k]=values[3*fields+k];
            report->characteristicInventory[k]=values[4*fields+k];
            report->inventoryChange[k]=values[3*fields+k]-values[2*fields+k];
        }
        const auto* budget=values+components;
        for(int k=0;k<GpuSteadySamplingReport::gasFluxFields;++k)report->gasBoundaryExchange[k]=budget[k];
        for(int k=0;k<6;++k){report->particleInjection[k]=budget[GpuSteadySamplingReport::gasFluxFields+k];
            report->particleTrackingExchange[k]=budget[GpuSteadySamplingReport::gasFluxFields+6+k];}
        return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
