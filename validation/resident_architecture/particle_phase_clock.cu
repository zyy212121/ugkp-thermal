#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "../../common/GpuParticlePhaseGraph.cuh"
struct State {
    State* deviceState=nullptr;
    cudaStream_t particleCaptureStream=nullptr;cudaGraph_t particleCaptureRoot=nullptr;
    ugkwp::ParticleResidentInterval particleResidentInterval{};
    ugkp::gpu::ParticlePhaseGraphs particlePhaseGraphs{};
    unsigned long long particleGraphEpoch=1;
    int preBaseDirectoryReady=0,splitPreDirectoryActive=0,csrTasksReady=0,csrPreparedDirectoryKind=0;
    double *source=nullptr,*destination=nullptr;
    double step=0,end=0,half=0,total=0;
};
static void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(1);}}
__global__ void update(State* s,double weight,double end){
    const double dt=ugkwp::particleResidentStepDt(*s,weight);
    s->step=dt;s->half=ugkwp::particleResidentStepDt(*s,weight*0.5);
    s->end=ugkwp::particleResidentEndTime(*s,end);
    s->destination[0]=s->source[0]+dt;s->total=s->destination[0];
    auto* old=s->source;s->source=s->destination;s->destination=old;
}
int main(){
    State state;check(cudaMalloc(&state.deviceState,sizeof(State)));
    check(cudaMalloc(&state.source,sizeof(double)));check(cudaMalloc(&state.destination,sizeof(double)));
    check(cudaMemset(state.source,0,sizeof(double)));check(cudaMemset(state.destination,0,sizeof(double)));
    check(cudaMemcpy(state.deviceState,&state,sizeof(state),cudaMemcpyHostToDevice));
    double end=0,total=0;int recordings=0,failures=0;
    for(int i=0;i<600;++i){
        const double dt=i%97==0?0.:i<250?0.0003:i<400?0.00017:0.03125;
        end+=dt;if(i==350)end=5.;total+=dt;
        if(i==450){ // Native configuration/state publication must not reset the graph clock.
            ++state.particleGraphEpoch;
            check(cudaMemcpy(state.deviceState,&state,sizeof(state),cudaMemcpyHostToDevice));
        }
        const int result=ugkp::gpu::launchParticlePhaseGraph(&state,dt,end,
            [&](double weight,double time){
                ++recordings;
                update<<<1,1,0,state.particleCaptureStream>>>(state.deviceState,weight,time);
                std::swap(state.source,state.destination); // Recording has host bookkeeping too.
                state.preBaseDirectoryReady=1;state.csrTasksReady=1;state.csrPreparedDirectoryKind=3;
                return int(cudaGetLastError()!=cudaSuccess);
            },[&]{std::swap(state.source,state.destination);},
            [&](const char* label,cudaError_t error){std::fprintf(stderr,"%s: %s\n",label,cudaGetErrorString(error));});
        if(result)return 2;
        State actual;check(cudaMemcpy(&actual,state.deviceState,sizeof(actual),cudaMemcpyDeviceToHost));
        const double half=dt*0.5;
        if(std::memcmp(&actual.step,&dt,sizeof(dt))||std::memcmp(&actual.end,&end,sizeof(end))
            ||std::memcmp(&actual.half,&half,sizeof(half))||actual.total!=total
            ||actual.source!=state.source||actual.destination!=state.destination
            ||actual.particleResidentInterval.active||state.preBaseDirectoryReady!=1
            ||state.csrTasksReady!=1||state.csrPreparedDirectoryKind!=3)++failures;
    }
    // Queue many replays without intermediate publication. Kernel arguments and
    // the device buffer/clock state must remain ordered as the host moves on.
    for(int i=0;i<256;++i){
        const double dt=i%31==0?0.:i%3==0?0.001:0.0007;end+=dt;total+=dt;
        const int result=ugkp::gpu::launchParticlePhaseGraph(&state,dt,end,
            [&](double weight,double time){
                ++recordings;update<<<1,1,0,state.particleCaptureStream>>>(state.deviceState,weight,time);
                std::swap(state.source,state.destination);state.preBaseDirectoryReady=1;
                state.csrTasksReady=1;state.csrPreparedDirectoryKind=3;return int(cudaGetLastError()!=cudaSuccess);
            },[&]{std::swap(state.source,state.destination);},
            [&](const char* label,cudaError_t error){std::fprintf(stderr,"%s: %s\n",label,cudaGetErrorString(error));});
        if(result)return 4;
    }
    State queued;check(cudaMemcpy(&queued,state.deviceState,sizeof(queued),cudaMemcpyDeviceToHost));
    if(queued.total!=total||queued.end!=end||queued.source!=state.source||queued.destination!=state.destination)++failures;
    check(cudaDeviceSynchronize());ugkp::gpu::releaseParticlePhaseGraphs(state.particlePhaseGraphs);
    check(cudaFree(state.source));check(cudaFree(state.destination));check(cudaFree(state.deviceState));
    std::printf("%s 856 physical intervals (256 queued), %d recordings, %d failures\n",failures?"FAIL":"PASS",recordings,failures);
    return failures?3:0;
}
