#pragma once
#include "gpu/ExecutionGraph.cuh"

namespace ugkp { namespace thermal {
namespace bound_particle_detail {
__global__ void prepare(DeviceState* state,const ugkwp::ParticleResidentInterval* interval,
    cudaGraphConditionalHandle initial,cudaGraphConditionalHandle retained){
    if(threadIdx.x||blockIdx.x)return;
    state->particleResidentInterval=*interval;
    const bool run=!diagnostics::hasFatal(state->diagnostics);
    cudaGraphSetConditional(initial,run&&!state->preBaseDirectoryReady);
    cudaGraphSetConditional(retained,run&&state->preBaseDirectoryReady);
}
__global__ void publishMirror(const DeviceState* state,BoundParticleMirror* mirror){
    if(threadIdx.x||blockIdx.x)return;
    *mirror={state->pCellId,state->preBaseDirectoryReady};
}
}

cudaGraphNode_t appendBoundParticlePhase(void* handle,cudaGraph_t graph,cudaGraphNode_t dependency,
    const ugkwp::ParticleResidentInterval* interval,cudaStream_t stream){
    auto* state=asState(handle);
    if(!state||!interval||!state->borrowedGasFields)
        throw std::invalid_argument("bound particle graph requires its owner, gas fields and device clock");
    using namespace gpu;using namespace executionGraphDetail;
    if(!state->particlesMayBePresent){
        cudaGraphNode_t node=nullptr;
        checked(cudaGraphAddEmptyNode(&node,graph,dependency?&dependency:nullptr,dependency?1:0));return node;
    }
    cudaGraphConditionalHandle initial,retained;
    checked(cudaGraphConditionalHandleCreate(&initial,graph,0,cudaGraphCondAssignDefault));
    checked(cudaGraphConditionalHandleCreate(&retained,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        bound_particle_detail::prepare<<<1,1,0,stream>>>(state->deviceState,interval,initial,retained);
    });
    std::vector<cudaGraphNode_t> branches;
    for(int base=0;base<2;++base){
        cudaGraph_t physical=nullptr;
        branches.push_back(addIf(graph,base?retained:initial,{selected},physical));
        const DeviceState saved=*state;
        try{
            state->particleCaptureStream=stream;state->particleCaptureRoot=graph;
            state->preBaseDirectoryReady=base;
#if GPU_DIRECTORY_HAS_BASE_ONLY
            state->csrTasksReady=base;state->csrPreparedDirectoryKind=base?int(HeavyDirectoryKind::baseOnly):-1;
#else
            state->csrTasksReady=0;state->csrPreparedDirectoryKind=-1;
#endif
            state->splitPreDirectoryActive=0;
            appendRecorded(physical,{},stream,[&]{
                if(submitResidentThermalParticles(state,1.,1.))throw std::runtime_error(lastError);
                finishParticlePhaseInterval<<<1,1,0,stream>>>(state->deviceState);
            });
        }catch(...){*state=saved;throw;}
        *state=saved;
    }
    cudaGraphNode_t completed=nullptr;
    checked(cudaGraphAddEmptyNode(&completed,graph,branches.data(),branches.size()));return completed;
}
int enqueueBoundParticleMirror(void* handle,BoundParticleMirror* output,cudaStream_t stream){
    auto* state=asState(handle);
    if(!state||!output){setLastErrorText("missing bound particle mirror storage");return 1;}
    bound_particle_detail::publishMirror<<<1,1,0,stream>>>(state->deviceState,output);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess){setLastError("publish bound particle mirror",error);return 1;}return 0;
}
int acceptBoundParticleMirror(void* handle,const BoundParticleMirror& mirror){
    auto* state=asState(handle);
    if(!state){setLastErrorText("missing bound particle mirror owner");return 1;}
    if(mirror.activeCells!=state->pCellId)swapParticleBufferPointersHost(state);
    state->preBaseDirectoryReady=mirror.baseReady;
    state->splitPreDirectoryActive=0;


#if GPU_DIRECTORY_HAS_BASE_ONLY
    state->csrTasksReady=mirror.baseReady;
    state->csrPreparedDirectoryKind=mirror.baseReady?int(HeavyDirectoryKind::baseOnly):-1;
#else
    state->csrTasksReady=0;state->csrPreparedDirectoryKind=-1;
#endif
    state->particleGeometryDirty=false;state->particlePhaseGraphs.submitted=false;return 0;
}
}}
