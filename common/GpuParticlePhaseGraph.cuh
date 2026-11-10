#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include "ParticleResidentInterval.H"
namespace ugkp { namespace gpu {
struct ParticlePhaseGraph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    cudaGraphNode_t intervalNode=nullptr;cudaKernelNodeParams intervalParameters{};
    ugkwp::ParticleResidentInterval parameters{};bool advances=false,sequenceAdvances=false;
    unsigned long long epoch=0;
    int tasksReady=0,preparedKind=0,baseReady=0,splitActive=0,injectionActive=0;
};
template<class State,class=void>struct HasThermalSplitDirectory:std::false_type{};
template<class State>struct HasThermalSplitDirectory<State,std::void_t<
    decltype(std::declval<State>().splitPreDirectoryActive)>>:std::true_type{};
template<class State>void saveParticleDirectoryMirror(const State& state,ParticlePhaseGraph& entry){
    entry.tasksReady=state.csrTasksReady;entry.preparedKind=state.csrPreparedDirectoryKind;
    entry.baseReady=state.preBaseDirectoryReady;
    if constexpr(HasThermalSplitDirectory<State>::value)entry.splitActive=state.splitPreDirectoryActive;
    else{entry.splitActive=state.useSplitPreDirectory;entry.injectionActive=state.preInjectionSegmentActive;}
}
template<class State>void restoreParticleDirectoryMirror(State& state,const ParticlePhaseGraph& entry){
    state.csrTasksReady=entry.tasksReady;state.csrPreparedDirectoryKind=entry.preparedKind;
    state.preBaseDirectoryReady=entry.baseReady;
    if constexpr(HasThermalSplitDirectory<State>::value)state.splitPreDirectoryActive=entry.splitActive;
    else{state.useSplitPreDirectory=entry.splitActive;state.preInjectionSegmentActive=entry.injectionActive;}
}
struct ParticlePhaseGraphs {
    ParticlePhaseGraph entry[4]{};
    ugkwp::ParticleResidentInterval* clock=nullptr;
    ugkwp::ParticleResidentInterval previous{};
    bool submitted=false,positionSequenceEnabled=false;
    // Last submitted sample. Configuration restores the last completed count.
    unsigned long long positionSequence=0;
};
inline cudaError_t releaseParticlePhaseGraph(ParticlePhaseGraph& entry){
    cudaError_t error=cudaSuccess;
    if(entry.executable)error=cudaGraphExecDestroy(entry.executable);
    if(entry.graph){auto next=cudaGraphDestroy(entry.graph);if(error==cudaSuccess)error=next;}
    entry={};return error;
}
inline void releaseParticlePhaseGraphs(ParticlePhaseGraphs& graphs){
    for(auto& entry:graphs.entry)releaseParticlePhaseGraph(entry);
    if(graphs.clock)cudaFree(graphs.clock);
    graphs={};
}
inline bool sameParticlePhysicalInterval(const ugkwp::ParticleResidentInterval& a,
    const ugkwp::ParticleResidentInterval& b){
    return std::memcmp(&a.dt,&b.dt,sizeof(double))==0
        &&std::memcmp(&a.end,&b.end,sizeof(double))==0&&a.active==b.active;
}
inline bool sameParticleInterval(const ugkwp::ParticleResidentInterval& a,
    const ugkwp::ParticleResidentInterval& b){
    return sameParticlePhysicalInterval(a,b)
        &&a.positionSequenceEnabled==b.positionSequenceEnabled
        &&a.positionSequence==b.positionSequence;
}
template<class State>
__global__ void publishParticlePhaseInterval(State* state,ugkwp::ParticleResidentInterval* clock,
    ugkwp::ParticleResidentInterval interval,bool advance,bool advanceSequence){
    if(!threadIdx.x&&!blockIdx.x){
        if(advance)interval.end=__dadd_rn(clock->end,interval.dt);
        if(advanceSequence)interval.positionSequence=clock->positionSequence+1;
        *clock=interval;state->particleResidentInterval=interval;
    }
}
template<class State>
__global__ void finishParticlePhaseInterval(State* state){
    if(!threadIdx.x&&!blockIdx.x)state->particleResidentInterval.active=false;
}
// Submit records the unchanged physical phase. Adapt publishes only its host
// pointer/directory mirror after replay; device pointers swap in the graph.
template<class State,class Submit,class Adapt,class Report>
int launchParticlePhaseGraph(State* state,double dt,double end,Submit&& submit,Adapt&& adapt,Report&& report){
    static_assert(std::is_trivially_copyable<State>::value,"particle graph recording snapshots require plain owner metadata");
    auto& owner=state->particlePhaseGraphs;
    auto& entry=owner.entry[(state->preBaseDirectoryReady?1:0)+(dt==0?2:0)];
    auto failed=[&](const char* label,cudaError_t error){report(label,error);return 1;};
    ugkwp::ParticleResidentInterval requested{dt,end,true};
    requested.positionSequenceEnabled=owner.positionSequenceEnabled;
    if(requested.positionSequenceEnabled)requested.positionSequence=owner.positionSequence+1;
    auto next=owner.previous;next.end=next.end+next.dt;
    bool advance=owner.submitted&&sameParticlePhysicalInterval(next,requested);
    bool advanceSequence=requested.positionSequenceEnabled&&owner.submitted
        &&owner.previous.positionSequenceEnabled
        &&owner.previous.positionSequence+1==requested.positionSequence;
    auto parameters=requested;
    if(advance)parameters.end=0; // ignored by device continuation; stable replay metadata
    if(advanceSequence)parameters.positionSequence=0; // independent of the physical clock
    if(!owner.clock){
        const auto error=cudaMalloc(reinterpret_cast<void**>(&owner.clock),sizeof(*owner.clock));
        if(error!=cudaSuccess)return failed("allocate particle graph clock",error);
    }
    if(!entry.executable||entry.epoch!=state->particleGraphEpoch){
        auto error=releaseParticlePhaseGraph(entry);
        if(error!=cudaSuccess)return failed("release particle phase graph",error);
        error=cudaGraphCreate(&entry.graph,0);
        if(error!=cudaSuccess)return failed("create particle phase graph",error);
        auto& kernel=entry.intervalParameters;
        kernel.func=reinterpret_cast<void*>(publishParticlePhaseInterval<State>);
        kernel.gridDim=dim3(1);kernel.blockDim=dim3(1);
        void* arguments[]={&state->deviceState,&owner.clock,&parameters,&advance,&advanceSequence};
        kernel.kernelParams=arguments;
        error=cudaGraphAddKernelNode(&entry.intervalNode,entry.graph,nullptr,0,&kernel);
        kernel.kernelParams=nullptr;
        if(error!=cudaSuccess)return failed("record particle phase interval",error);
        cudaStream_t stream=nullptr;
        error=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
        if(error!=cudaSuccess)return failed("create particle phase recording stream",error);
        error=cudaStreamBeginCaptureToGraph(stream,entry.graph,&entry.intervalNode,nullptr,1,cudaStreamCaptureModeThreadLocal);
        if(error!=cudaSuccess){cudaStreamDestroy(stream);return failed("begin particle phase recording",error);}
        const State saved=*state;
        state->particleCaptureStream=stream;state->particleCaptureRoot=entry.graph;
        int status=submit(dt==0?0.:1.,1.);
        if(!status){
            finishParticlePhaseInterval<<<1,1,0,stream>>>(state->deviceState);
            const auto launch=cudaGetLastError();
            if(launch!=cudaSuccess){report("finish particle phase interval",launch);status=1;}
        }
        ParticlePhaseGraph mirror;saveParticleDirectoryMirror(*state,mirror);
        *state=saved;
        error=cudaStreamEndCapture(stream,&entry.graph);
        const auto destroyed=cudaStreamDestroy(stream);
        if(status)return status;
        if(error!=cudaSuccess)return failed("end particle phase recording",error);
        if(destroyed!=cudaSuccess)return failed("destroy particle phase recording stream",destroyed);
        error=cudaGraphInstantiateWithFlags(&entry.executable,entry.graph,0);
        if(error!=cudaSuccess)return failed("instantiate particle phase graph",error);
        entry.tasksReady=mirror.tasksReady;entry.preparedKind=mirror.preparedKind;
        entry.baseReady=mirror.baseReady;entry.splitActive=mirror.splitActive;entry.injectionActive=mirror.injectionActive;
        entry.parameters=parameters;entry.advances=advance;entry.sequenceAdvances=advanceSequence;entry.epoch=state->particleGraphEpoch;
    }else if(entry.advances!=advance||entry.sequenceAdvances!=advanceSequence
        ||!sameParticleInterval(entry.parameters,parameters)){
        void* arguments[]={&state->deviceState,&owner.clock,&parameters,&advance,&advanceSequence};
        auto kernel=entry.intervalParameters;kernel.kernelParams=arguments;
        const auto error=cudaGraphExecKernelNodeSetParams(entry.executable,entry.intervalNode,&kernel);
        if(error!=cudaSuccess)return failed("update particle phase interval",error);
        entry.parameters=parameters;entry.advances=advance;entry.sequenceAdvances=advanceSequence;
    }
    const auto error=cudaGraphLaunch(entry.executable,nullptr);
    if(error!=cudaSuccess)return failed("launch particle phase graph",error);
    adapt();
    restoreParticleDirectoryMirror(*state,entry);
    owner.previous=requested;owner.submitted=true;
    if(requested.positionSequenceEnabled)owner.positionSequence=requested.positionSequence;
    return 0;
}
}}
