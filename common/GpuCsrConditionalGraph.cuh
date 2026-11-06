#pragma once
#define UGKP_RESIDENT_CSR_CHOICE 1
#include "GpuCsrConditionalGraph.H"
// One device choice governs the whole reduction path; the other conditional
// body does not execute. No floating-point operation or RNG draw is added.
template<class State>
__global__ void publishCsrGraphChoice(State* s,cudaGraphConditionalHandle heavy,cudaGraphConditionalHandle light) {
    if(threadIdx.x||blockIdx.x)return;
    const int active=s->csrAutomaticRuntime->active;
    s->csrHeavyReductionActive=active;s->csrHeavyReductionEnabled=active;
    cudaGraphSetConditional(heavy,active!=0);cudaGraphSetConditional(light,active==0);
}
template<class State>
__global__ void setCsrAutomaticTimeStep(State* s,double dt) {
    if(!threadIdx.x&&!blockIdx.x)s->csrAutomaticRuntime->timeStep=dt;
}
template<class State,class Submit>
int launchCsrConditional(State* s,int kind,int block,int directory,int variant,Submit&& submit) {
    if(s->csrHeavyReductionMode!=2)return submit(nullptr,s->csrHeavyReductionEnabled!=0);
    ugkp::gpu::CsrChoiceKey key;
    key.state=s->deviceState;key.temporary=s->cellScanTempStorage;key.temporaryBytes=s->cellScanTempBytes;
    key.counts=s->csrCellTaskCount;key.offsets=s->csrCellTaskOffset;key.queue=s->csrHeavyTaskCursor;key.runtime=s->csrAutomaticRuntime;
    key.heavyTasks=s->csrHeavyTaskCount;key.heavyCells=s->csrHeavyCellCount;
    key.cells=s->nCells;key.block=block;key.workers=s->csrHeavyWorkerGrid;key.multiprocessors=s->multiprocessorCount;
    key.directory=directory;key.variant=variant;
    auto& entry=s->csrChoiceGraphs.entry[kind];
    auto failed=[&](const char* where,cudaError_t error){setLastError(where,error);ugkp::gpu::releaseCsrChoiceGraph(entry);return 1;};
    if(!entry.executable||!(entry.key==key)) {
        auto error=ugkp::gpu::releaseCsrChoiceGraph(entry);if(error!=cudaSuccess)return failed("release CSR choice graph",error);
        error=cudaGraphCreate(&entry.graph,0);if(error!=cudaSuccess)return failed("create CSR choice graph",error);
        cudaGraphConditionalHandle handles[2];
        for(auto& handle:handles){error=cudaGraphConditionalHandleCreate(&handle,entry.graph,0,cudaGraphCondAssignDefault);
            if(error!=cudaSuccess)return failed("create CSR conditional handle",error);}
        cudaKernelNodeParams publication{};publication.func=reinterpret_cast<void*>(publishCsrGraphChoice<State>);
        publication.gridDim=dim3(1);publication.blockDim=dim3(1);
        void* arguments[]={&s->deviceState,&handles[0],&handles[1]};publication.kernelParams=arguments;
        cudaGraphNode_t decision;
        error=cudaGraphAddKernelNode(&decision,entry.graph,nullptr,0,&publication);
        if(error!=cudaSuccess)return failed("add CSR choice publication",error);
        cudaStream_t stream=nullptr;error=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
        if(error!=cudaSuccess)return failed("create CSR choice stream",error);
        for(int branch=0;branch<2;++branch) {
            cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
            parameters.conditional.handle=handles[branch];parameters.conditional.type=cudaGraphCondTypeIf;parameters.conditional.size=1;
            cudaGraphNode_t node;
#if CUDART_VERSION >= 13000
            error=cudaGraphAddNode(&node,entry.graph,&decision,nullptr,1,&parameters);
#else
            error=cudaGraphAddNode(&node,entry.graph,&decision,1,&parameters);
#endif
            if(error!=cudaSuccess){cudaStreamDestroy(stream);return failed("add CSR conditional body",error);}
            auto body=parameters.conditional.phGraph_out[0];
            error=cudaStreamBeginCaptureToGraph(stream,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal);
            if(error!=cudaSuccess){cudaStreamDestroy(stream);return failed("begin CSR branch capture",error);}
            const int status=submit(stream,branch==0);
            error=cudaStreamEndCapture(stream,&body);
            if(status||error!=cudaSuccess){cudaStreamDestroy(stream);return failed("capture CSR branch",error==cudaSuccess?cudaErrorUnknown:error);}
        }
        error=cudaStreamDestroy(stream);if(error!=cudaSuccess)return failed("release CSR choice stream",error);
        error=cudaGraphInstantiateWithFlags(&entry.executable,entry.graph,0);
        if(error!=cudaSuccess)return failed("instantiate CSR choice graph",error);
        entry.key=key;
    }
    const auto error=cudaGraphLaunch(entry.executable,nullptr);
    if(error!=cudaSuccess)return failed("launch CSR choice graph",error);
    return 0;
}
