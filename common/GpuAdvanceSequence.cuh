#pragma once



#include "gpu/AdvanceState.cuh"

#ifndef UGKP_ADVANCE_TRIAL_POLICY
#define UGKP_ADVANCE_TRIAL_POLICY SharedGasTrialPolicy
#endif
namespace {
template<class State> void prepareAdvanceDirectoryMirror(State& state,int base){
    state.preBaseDirectoryReady=base;
#if GPU_DIRECTORY_HAS_BASE_ONLY
    state.csrTasksReady=base?1:0;
    state.csrPreparedDirectoryKind=base?int(HeavyDirectoryKind::baseOnly):-1;
#else


    state.csrTasksReady=0;state.csrPreparedDirectoryKind=-1;
#endif
    if constexpr(ugkp::gpu::HasThermalSplitDirectory<State>::value)state.splitPreDirectoryActive=0;
    else{state.useSplitPreDirectory=0;state.preInjectionSegmentActive=0;}
}
template<class State> void collectAdvanceDirectoryMirror(State& state,const ugkp::gpu::AdvanceGraphStorage& owner){
    state.csrTasksReady=owner.tasksReady;state.csrPreparedDirectoryKind=owner.preparedKind;
    if constexpr(ugkp::gpu::HasThermalSplitDirectory<State>::value)state.splitPreDirectoryActive=owner.splitActive;
    else{state.useSplitPreDirectory=owner.splitActive;state.preInjectionSegmentActive=owner.injectionActive;}
}
void releaseAdvanceSequence(DeviceState* state){delete state->advanceGraph;state->advanceGraph=nullptr;}
void buildAdvanceSequence(DeviceState* state,bool withCourant){
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    auto& owner=*state->advanceGraph;
    struct Restore {DeviceState* target;DeviceState saved;~Restore(){*target=saved;}} restore{state,*state};
    state->advanceRecording=true;state->gasCaptureStream=owner.stream;
    owner.clearGraph();checked(cudaGraphCreate(&owner.graph,0));
    const bool split=state->particlesMayBePresent && state->csrCellLocalPathEnabled && !owner.pureGas;
    cudaGraphConditionalHandle loop,initial,retained{};
    for(auto handle:{&loop,&initial})
        checked(cudaGraphConditionalHandleCreate(handle,owner.graph,0,cudaGraphCondAssignDefault));
    if(split)checked(cudaGraphConditionalHandleCreate(&retained,owner.graph,0,cudaGraphCondAssignDefault));
    const auto start=appendRecorded(owner.graph,{},owner.stream,[&]{
        checked(cudaMemcpyAsync(owner.request,owner.hostRequest,sizeof(AdvanceRequest),cudaMemcpyHostToDevice,owner.stream));
        beginAdvanceSubmission<<<1,1,0,owner.stream>>>(owner.state,owner.request,state->diagnostics,loop);
    });
    cudaGraph_t body=nullptr;const auto advanced=addWhile(owner.graph,loop,{start},body);
    std::vector<cudaGraphNode_t> selectionInputs;
    if(withCourant){
        cudaGraphConditionalHandle refresh;
        checked(cudaGraphConditionalHandleCreate(&refresh,owner.graph,0,cudaGraphCondAssignDefault));
        const auto prepared=appendRecorded(body,{},owner.stream,[&]{
            prepareAdvanceCourant<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,owner.request,refresh);
        });
        cudaGraph_t cfl=nullptr;
        selectionInputs.push_back(addIf(body,refresh,{prepared},cfl));
        appendRecorded(cfl,{},owner.stream,[&]{
            if(computeGasCurrentCourantField(state,1.0,0.0,state->fixedCellBlockThreads,state->fixedFaceBlockThreads,owner.stream))
                throw std::runtime_error(lastError);
            advanceMaximum(state->gasFluxPositivityScale,state->nCells,state->reductionBlockThreads,owner);
            if(state->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy&&UGKP_ADVANCE_TRIAL_POLICY::validate(state))
                throw std::runtime_error(lastError);
        });
    }
    auto selected=appendRecorded(body,selectionInputs,owner.stream,[&]{
        selectAdvanceStep<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,owner.request,
            owner.maximumWorkspace+owner.maximumParts,initial,retained,split);
    });
#ifdef UGKWP_STEADY_APPLICATION
    selected=appendSteadyAdvancePreparation(state,owner,body,selected);
    selected=appendRecorded(body,{selected},owner.stream,[&]{
        enterSteadyPhysicalBranch<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,initial,retained,split);
    });
#endif
    std::vector<cudaGraphNode_t> branches;
    for(int base=0;base<(split?2:1);++base){
        const auto segment=base?retained:initial;
        cudaGraph_t physical=nullptr;
#ifdef UGKWP_STEADY_APPLICATION
        branches.push_back(addIf(body,segment,{selected},physical));
#else
        branches.push_back(addWhile(body,segment,{selected},physical));
#endif
        const DeviceState saved=*state;
        try{
            state->advanceRecording=true;state->gasGraphMode=0;
            state->gasCaptureStream=owner.stream;state->particleCaptureStream=owner.stream;
            state->particleCaptureRoot=owner.graph;prepareAdvanceDirectoryMirror(*state,base);
            ugkwp::invalidateGasBoundaryLayerPreparation(*state);
            appendRecorded(physical,{},owner.stream,[&]{
                const bool mixtureOnly=owner.pureGas&&state->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy;
                const int result=owner.pureGas
                    ?ugkwpGpuResidentStrictAdvanceGasOnly(state,1.0,mixtureOnly?0.0:1.0)
                    :ugkwpGpuResidentStrictAdvance(state,1.0,1.0);
                if(result)throw std::runtime_error(lastError);
#ifndef UGKWP_STEADY_APPLICATION
                finishPhysicalSegmentStep<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,owner.request,
                    owner.maximumWorkspace+owner.maximumParts,segment,split,base,bool(owner.extension),owner.deviceProgress);
                checked(cudaGetLastError());
#endif
            });
            ParticlePhaseGraph mirror;saveParticleDirectoryMirror(*state,mirror);
            owner.tasksReady=mirror.tasksReady;owner.preparedKind=mirror.preparedKind;
            owner.baseReady=mirror.baseReady;owner.splitActive=mirror.splitActive;owner.injectionActive=mirror.injectionActive;
        }catch(...){*state=saved;throw;}
        *state=saved;
    }
    auto physicalDone=joinGraphTails(body,branches);
#ifdef UGKWP_STEADY_APPLICATION
    appendSteadyAcceptance(state,owner,body,physicalDone,loop);
#else
    if(owner.extension)physicalDone=owner.extension->appendTo(body,physicalDone,owner.state,owner.request,owner.stream);
    appendRecorded(body,{physicalDone},owner.stream,[&]{
        completeThermalSegment<<<1,1,0,owner.stream>>>(state->deviceState,owner.state,owner.request,loop,owner.deviceProgress);
    });
#endif
    appendRecorded(owner.graph,{advanced},owner.stream,[&]{
        publishAdvanceReceipt<<<1,128,0,owner.stream>>>(state->deviceState,owner.state,owner.receipt);
        checked(cudaMemcpyAsync(owner.hostReceipt,owner.receipt,sizeof(AdvanceReceipt),cudaMemcpyDeviceToHost,owner.stream));
    });
    checked(cudaGraphInstantiateWithFlags(&owner.executable,owner.graph,0));
    owner.epoch=state->particleGraphEpoch;
    owner.includesCourant=withCourant;
}
}
extern "C" int ugkwpGpuResidentStrictConfigureAdvance(void* handle,const ugkp::gpu::AdvanceState* initial,int pureGas){
    auto* state=asState(handle);
    if(validateState(state,"configure device advance")||!initial)return 1;
    try{
        using namespace ugkp::gpu;
        if(state->advanceGraph&&state->advanceGraph->pending)
            throw std::logic_error("collect the active advance before reconfiguration");
        releaseAdvanceSequence(state);
        auto owner=std::make_unique<AdvanceGraphStorage>();owner->pureGas=pureGas!=0;
        owner->initialise((state->nCells+state->reductionBlockThreads-1)/state->reductionBlockThreads);
        executionGraphDetail::checked(cudaEventRecord(owner->completed,nullptr));
        executionGraphDetail::checked(cudaStreamWaitEvent(owner->stream,owner->completed,0));
        auto clock=*initial;
        if(clock.clock==AdvanceClock::physical)clock.clockDt=clock.dt;
        owner->acceptedStepsAtStart=clock.steps;owner->acceptedTime=clock.time;owner->nominalDt=clock.dt;
        {PinnedTransferQueue::Operation upload(owner->transfers);owner->transfers.upload(owner->state,&clock,sizeof(clock));}
        state->advanceGraph=owner.release();return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictEnqueueAdvance(void* handle,const ugkp::gpu::AdvanceRequest* request){
    auto* state=asState(handle);
    if(validateState(state,"submit device advance")||!request)return 1;
    try{
        using namespace ugkp::gpu;using executionGraphDetail::checked;
        if(!state->advanceGraph)throw std::logic_error("device advance must be configured before submission");
        auto& owner=*state->advanceGraph;
        if(owner.pending)throw std::logic_error("device advance result lease is still active");
        if(!(request->boundary==request->boundary)||!request->maximumSteps||request->courantInterval<1)
            throw std::invalid_argument("invalid device advance boundary or step budget");
        if(!owner.executable||owner.epoch!=state->particleGraphEpoch||(request->adjustTimeStep&&!owner.includesCourant))
            buildAdvanceSequence(state,request->adjustTimeStep||owner.includesCourant);
        auto bounded=*request;
        if(bounded.progressInterval)
            bounded.maximumSteps=std::min(bounded.maximumSteps,std::uint64_t(advanceProgressCapacity)*bounded.progressInterval);
        if(bounded.watchControlFiles)
            bounded.maximumSteps=std::min<std::uint64_t>(bounded.maximumSteps,advanceProgressCapacity);
        *owner.hostRequest=bounded;
        owner.pending=true;
        if(owner.extension)owner.extension->orderPreparedInputs(owner.stream);
        if(owner.mayReachThermal(bounded)){
            owner.task.start(owner.stream,owner.executable,[&owner]{
                if(!owner.extension->servicePendingCapacity())return false;
                resumeAdvanceCapacity<<<1,1,0,owner.stream>>>(owner.state);
                executionGraphDetail::checked(cudaGetLastError());return true;
            },[&owner]{
                owner.extension->acceptExternalExecution(owner.stream);
                executionGraphDetail::checked(cudaEventRecord(owner.completed,owner.stream));
            });
            owner.extension->acceptExternalCompletion(owner.task.result());
        }else{
            checked(cudaGraphLaunch(owner.executable,owner.stream));
            checked(cudaEventRecord(owner.completed,owner.stream));
        }
        return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
extern "C" int ugkwpGpuResidentStrictCollectAdvanceProgress(void* handle,ugkp::gpu::AdvanceState* result,
    ugkp::gpu::AdvanceProgress* progress,bool (*controlsChanged)(const void*),const void* observer){
    if(progress)progress->count=0;
    auto* state=asState(handle);
    if(!state||!result||!state->advanceGraph){setLastErrorText("missing device advance result owner");return 1;}
    try{
        using namespace ugkp::gpu;
        auto& owner=*state->advanceGraph;
        if(!owner.pending)throw std::logic_error("no submitted device advance to collect");
        owner.task.wait();
        executionGraphDetail::checked(cudaEventSynchronize(owner.completed));
        const auto& receipt=*owner.hostReceipt;
        *result=receipt.state;
        if(progress && result->progressCount){
            std::copy_n(owner.hostProgress,result->progressCount,progress->points);
            progress->count=result->progressCount;
        }
        owner.pending=false;
        owner.acceptedStepsAtStart=result->steps;
        if(controlsChanged && owner.hostRequest->watchControlFiles)controlsChanged(observer);
        owner.acceptedTime=result->time;
        owner.nominalDt=result->dt;
        if(owner.reportBatches){
            ++owner.batches;owner.acceptedSteps+=result->submissionSteps;
            if(result->submissionSteps==1)++owner.singleStepBatches;
            owner.longestBatch=std::max(owner.longestBatch,result->submissionSteps);
        }
        if(receipt.activeCells!=state->pCellId)swapParticleBufferPointersHost(state);
        state->preBaseDirectoryReady=receipt.baseReady;collectAdvanceDirectoryMirror(*state,owner);
        state->particlePhaseGraphs.positionSequence=result->positionSequence;
        state->particlePhaseGraphs.submitted=false;state->gasGraphIntervalSubmitted=false;
        ugkwp::invalidateGasBoundaryLayerPreparation(*state);
        state->diagnosticOwner->consumeCompletedSnapshot(receipt.diagnostics);
        return state->diagnosticOwner->dispatchPending()||result->stop==AdvanceStop::fatal?1:0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}

extern "C" int ugkwpGpuResidentStrictCollectAdvance(void* handle,ugkp::gpu::AdvanceState* result){
    return ugkwpGpuResidentStrictCollectAdvanceProgress(handle,result,nullptr,nullptr,nullptr);
}

int ugkp::gpu::bindAdvanceExtension(void* handle,std::shared_ptr<AdvanceExtension> extension){
    auto* state=asState(handle);
    if(validateState(state,"bind advance dependency")||!state->advanceGraph)return 1;
    try{
        auto& owner=*state->advanceGraph;
        if(owner.pending)throw std::logic_error("advance dependency still belongs to active work");
        owner.task.wait();owner.clearGraph();owner.extension=std::move(extension);return 0;
    }catch(const std::exception& error){setLastErrorText(error.what());return 1;}
}
#undef UGKP_ADVANCE_TRIAL_POLICY
extern "C" void ugkwpGpuResidentStrictReleaseAdvance(void* handle) noexcept{
    auto* state=asState(handle);if(state)releaseAdvanceSequence(state);
}
