#pragma once
#define UGKP_RESIDENT_CSR_CHOICE 1
#include "GpuCsrConditionalGraph.H"
#include "GpuParticleExecutionContext.H"
#include "ParticleResidentInterval.H"
// One device choice governs the whole reduction path; the other conditional
// body does not execute. No floating-point operation or RNG draw is added.
template<class State>
__global__ void publishCsrGraphChoice(State* s,cudaGraphConditionalHandle heavy,cudaGraphConditionalHandle light,bool inspectionOnly) {
    if(threadIdx.x||blockIdx.x)return;
    const int active=s->csrAutomaticRuntime->active;
    s->csrHeavyReductionActive=active;s->csrHeavyReductionEnabled=active;
    const bool run=!inspectionOnly||s->csrAutomaticRuntime->inspect;
    cudaGraphSetConditional(heavy,run&&active!=0);cudaGraphSetConditional(light,run&&active==0);
}
template<class State>
__global__ void setCsrAutomaticTimeStep(State* s,double dt) {
    if(!threadIdx.x&&!blockIdx.x)s->csrAutomaticRuntime->timeStep=ugkwp::particleResidentStepDt(*s,dt);
}
// Add conditional bodies directly to their owning graph. A conditional graph
// cannot be cloned as a child of a captured whole-particle phase.
template<class State,class Submit>
cudaError_t appendCsrChoiceNodes(State* s,cudaGraph_t root,cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,std::size_t dependencyCount,
    cudaGraphNode_t (&joined)[2],Submit&& submit,bool inspectionOnly){
    cudaGraphConditionalHandle handles[2];
    for(auto& handle:handles){
        const auto error=cudaGraphConditionalHandleCreate(&handle,root,0,cudaGraphCondAssignDefault);
        if(error!=cudaSuccess)return error;
    }
    cudaKernelNodeParams publication{};publication.func=reinterpret_cast<void*>(publishCsrGraphChoice<State>);
    publication.gridDim=dim3(1);publication.blockDim=dim3(1);
    void* arguments[]={&s->deviceState,&handles[0],&handles[1],&inspectionOnly};publication.kernelParams=arguments;
    cudaGraphNode_t decision;
    auto error=cudaGraphAddKernelNode(&decision,graph,dependencies,dependencyCount,&publication);
    if(error!=cudaSuccess)return error;
    cudaStream_t stream=nullptr;error=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    if(error!=cudaSuccess)return error;
    for(int branch=0;branch<2;++branch){
        cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
        parameters.conditional.handle=handles[branch];parameters.conditional.type=cudaGraphCondTypeIf;parameters.conditional.size=1;
#if CUDART_VERSION >= 13000
        error=cudaGraphAddNode(&joined[branch],graph,&decision,nullptr,1,&parameters);
#else
        error=cudaGraphAddNode(&joined[branch],graph,&decision,1,&parameters);
#endif
        if(error!=cudaSuccess)break;
        auto body=parameters.conditional.phGraph_out[0];
        error=cudaStreamBeginCaptureToGraph(stream,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);
        if(error!=cudaSuccess)break;
        const int status=submit(stream,branch==0);
        error=cudaStreamEndCapture(stream,&body);
        if(status||error!=cudaSuccess){if(error==cudaSuccess)error=cudaErrorUnknown;break;}
    }
    const auto destroyed=cudaStreamDestroy(stream);
    return error==cudaSuccess?destroyed:error;
}
template<class State,class Submit>
int launchCsrConditional(State* s,int kind,int block,int directory,int variant,Submit&& submit,bool inspectionOnly=false) {
    const auto recording=ugkp::gpu::particleRecordingStream(s);
    if(s->csrHeavyReductionMode!=2)return submit(recording,s->csrHeavyReductionEnabled!=0);
    if(recording){
        cudaStreamCaptureStatus status;cudaGraph_t graph=nullptr;unsigned long long id=0;
        const cudaGraphNode_t* dependencies=nullptr;std::size_t count=0;
#if CUDART_VERSION >= 13000
        auto error=cudaStreamGetCaptureInfo(recording,&status,&id,&graph,&dependencies,nullptr,&count);
#else
        auto error=cudaStreamGetCaptureInfo_v2(recording,&status,&id,&graph,&dependencies,&count);
#endif
        if(error==cudaSuccess&&status!=cudaStreamCaptureStatusActive)error=cudaErrorStreamCaptureInvalidated;
        cudaGraphNode_t joined[2];
        if(error==cudaSuccess)error=appendCsrChoiceNodes(s,ugkp::gpu::particleRecordingRoot(s),graph,
            dependencies,count,joined,submit,inspectionOnly);
#if CUDART_VERSION >= 13000
        if(error==cudaSuccess)error=cudaStreamUpdateCaptureDependencies(recording,joined,nullptr,2,cudaStreamSetCaptureDependencies);
#else
        if(error==cudaSuccess)error=cudaStreamUpdateCaptureDependencies(recording,joined,2,cudaStreamSetCaptureDependencies);
#endif
        if(error!=cudaSuccess){setLastError("append resident CSR choice",error);return 1;}
        return 0;
    }
    ugkp::gpu::CsrChoiceKey key;
    key.state=s->deviceState;key.temporary=s->cellScanTempStorage;key.temporaryBytes=s->cellScanTempBytes;
    key.counts=s->csrCellTaskCount;key.offsets=s->csrCellTaskOffset;key.queue=s->csrHeavyTaskCursor;key.runtime=s->csrAutomaticRuntime;
    key.heavyTasks=s->csrHeavyTaskCount;key.heavyCells=s->csrHeavyCellCount;
    key.cells=s->nCells;key.block=block;key.workers=s->csrHeavyWorkerGrid;key.multiprocessors=s->multiprocessorCount;
    key.directory=directory;key.variant=variant;key.inspectionOnly=inspectionOnly;
    auto& entry=s->csrChoiceGraphs.entry[kind];
    auto failed=[&](const char* where,cudaError_t error){setLastError(where,error);ugkp::gpu::releaseCsrChoiceGraph(entry);return 1;};
    if(!entry.executable||!(entry.key==key)) {
        auto error=ugkp::gpu::releaseCsrChoiceGraph(entry);if(error!=cudaSuccess)return failed("release CSR choice graph",error);
        error=cudaGraphCreate(&entry.graph,0);if(error!=cudaSuccess)return failed("create CSR choice graph",error);
        cudaGraphNode_t joined[2];
        error=appendCsrChoiceNodes(s,entry.graph,entry.graph,nullptr,0,joined,submit,inspectionOnly);
        if(error!=cudaSuccess)return failed("construct CSR conditional bodies",error);
        error=cudaGraphInstantiateWithFlags(&entry.executable,entry.graph,0);
        if(error!=cudaSuccess)return failed("instantiate CSR choice graph",error);
        entry.key=key;
    }
    const auto error=cudaGraphLaunch(entry.executable,nullptr);
    if(error!=cudaSuccess)return failed("launch CSR choice graph",error);
    return 0;
}
