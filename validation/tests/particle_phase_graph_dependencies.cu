// Production graph owner probe; execute only as a scheduled checkpoint batch.
#include <cuda_runtime.h>
#include <type_traits>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "GpuParticlePhaseGraph.cuh"
#ifdef UGKP_TEST_FP32
using Real=float;
#else
using Real=double;
#endif
struct State {
    State* deviceState=nullptr;
    ugkp::gpu::ParticlePhaseGraphs particlePhaseGraphs{};
    ugkwp::ParticleResidentInterval particleResidentInterval{};
    cudaStream_t particleCaptureStream=nullptr;
    cudaGraph_t particleCaptureRoot=nullptr;
    unsigned long long particleGraphEpoch=1;
    int preBaseDirectoryReady=0,csrTasksReady=0,csrPreparedDirectoryKind=0;
    int useSplitPreDirectory=0,preInjectionSegmentActive=0;
};
static void checked(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);}}
__global__ void observe(State* s,Real* output,int count,int model){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=Real(s->particleResidentInterval.end+2*s->particleResidentInterval.dt+model);
}
int main(){
    State state;checked(cudaMalloc(&state.deviceState,sizeof(State)));
    Real *first=nullptr,*second=nullptr;checked(cudaMallocManaged(&first,128*sizeof(Real)));
    checked(cudaMallocManaged(&second,128*sizeof(Real)));
    checked(cudaMemcpy(state.deviceState,&state,sizeof(State),cudaMemcpyHostToDevice));
    Real* target=first;int count=64,model=0,captures=0;
    auto run=[&](double dt,double end){
        auto submit=[&](double,double){++captures;
            observe<<<(count+31)/32,32,0,state.particleCaptureStream>>>(state.deviceState,target,count,model);
            return cudaGetLastError()==cudaSuccess?0:1;};
        auto report=[](const char* w,cudaError_t e){std::fprintf(stderr,"%s %s\n",w,cudaGetErrorString(e));};
        if(ugkp::gpu::launchParticlePhaseGraph(&state,dt,end,submit,[]{},report))std::exit(3);
        checked(cudaDeviceSynchronize());
        const Real expected=Real(end+2*dt+model);
        for(int i=0;i<count;++i)if(std::abs(double(target[i]-expected))>1e-6){std::fprintf(stderr,"value mismatch\n");std::exit(4);}
    };
    // A header publication with unchanged inputs keeps the capture epoch.
    for(int i=1;i<=8;++i){
        checked(cudaMemcpy(state.deviceState,&state,sizeof(State),cudaMemcpyHostToDevice));
        run(.125,i*.125);
    }
    if(captures!=1)return 5;
    run(.25,3.); // Discontinuous interval must update, not recapture.
    if(captures!=1)return 6;
    // Configuration/owner operations continue invalidating real dependencies.
    target=second;++state.particleGraphEpoch;run(.25,3.25);
    count=128;++state.particleGraphEpoch;run(.25,3.5);
    model=7;++state.particleGraphEpoch;run(.25,3.75);
    if(captures!=4)return 7;
    run(0.,3.75); // Zero-duration branch has a separate cache slot.
    if(captures!=5)return 8;
    std::printf("PASS precision=%zu captures=%d",sizeof(Real)*8,captures);
#if defined(UGKP_DEVELOPMENT_PROBES) && UGKP_DEVELOPMENT_PROBES
    std::printf(" ownerCaptures=%llu intervalUpdates=%llu launches=%llu",
        state.particlePhaseGraphs.captureCount,state.particlePhaseGraphs.intervalUpdateCount,
        state.particlePhaseGraphs.launchCount);
    if(state.particlePhaseGraphs.captureCount!=5||state.particlePhaseGraphs.launchCount!=13
        ||state.particlePhaseGraphs.intervalUpdateCount<2)return 9;
#endif
    std::puts("");
    ugkp::gpu::releaseParticlePhaseGraphs(state.particlePhaseGraphs);
    checked(cudaFree(first));checked(cudaFree(second));checked(cudaFree(state.deviceState));
    return 0;
}
