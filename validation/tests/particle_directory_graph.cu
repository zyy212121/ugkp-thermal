// Native probe for the production graph owner and unchanged count/scatter kernels.
// The split-injection application pipeline is covered by application regressions.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "GpuParticleDirectoryGraph.H"
static void checked(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}}
static void setLastError(const char* where,cudaError_t e){std::fprintf(stderr,"%s: %s\n",where,cudaGetErrorString(e));}
struct DeviceState {
    ugkp::gpu::ParticleDirectoryGraphs particleDirectoryGraphs{};
    DeviceState* deviceState=nullptr;
    void* cellScanTempStorage=nullptr;
    size_t cellScanTempBytes=0;
    int nCells=7,particleCapacity=257,particleWorkGrid=3,particleBlockThreads=32;
    int csrWarpAggregatedBinning=0,nBoundarySources=0;
    int *cellParticleCount=nullptr,*cellParticleOffset=nullptr,*cellParticleWrite=nullptr;
    int *particleCountDevice=nullptr,*pCellId=nullptr,*pStatus=nullptr,*sortedParticleIndex=nullptr;
};
__device__ int clampRange(int v,int low,int high){return min(max(v,low),high);}
#include "operators/clearParticleCellBinsKernel.cuh"
#include "operators/countParticlesByCellKernel.cuh"
#include "GpuParticleDirectoryGraph.cuh"

static int submitFull(DeviceState* s,bool survivor,cudaStream_t stream) {
    const int block=32,grid=(s->nCells+1+block-1)/block;
    clearParticleCellBinsKernel<<<grid,block,0,stream>>>(s->deviceState);
    if(s->csrWarpAggregatedBinning) {
        if(survivor)countParticlesByCellKernel<true,true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
        else countParticlesByCellKernel<true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
    } else {
        if(survivor)countParticlesByCellKernel<false,true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
        else countParticlesByCellKernel<false><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
    }
    checked(cudaGetLastError());
    checked(cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage,s->cellScanTempBytes,s->cellParticleCount,s->cellParticleOffset,s->nCells+1,stream));
    initialiseParticleCellWritesKernel<<<grid,block,0,stream>>>(s->deviceState);
    if(s->csrWarpAggregatedBinning) {
        if(survivor)scatterParticlesByCellKernel<true,true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
        else scatterParticlesByCellKernel<true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
    } else {
        if(survivor)scatterParticlesByCellKernel<false,true><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
        else scatterParticlesByCellKernel<false><<<s->particleWorkGrid,s->particleBlockThreads,0,stream>>>(s->deviceState);
    }
    checked(cudaGetLastError());return 0;
}
__global__ void producer(DeviceState* sp,int seed,int count) {
    auto& s=*sp;const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0)*s.particleCountDevice=count;
    if(i<s.particleCapacity){s.pCellId[i]=(i*13+seed)%(s.nCells+2)-1;s.pStatus[i]=(i+seed)%3;}
}
__global__ void consumer(DeviceState* sp,int* total){if(threadIdx.x==0)*total=sp->cellParticleOffset[sp->nCells];}
static void allocateInt(int*& p,int n){checked(cudaMallocManaged(&p,n*sizeof(int)));}
static void publish(DeviceState& s){checked(cudaMemcpy(s.deviceState,&s,sizeof(s),cudaMemcpyHostToDevice));}
static bool verify(const DeviceState& s,int seed,int count,bool survivor,int total) {
    int expectedTotal=0;
    for(int c=0;c<s.nCells;++c) {
        std::vector<int> expected,actual;
        for(int i=0;i<count;++i)if((i*13+seed)%(s.nCells+2)-1==c&&
            (survivor?(i+seed)%3==1:(i+seed)%3!=0))expected.push_back(i);
        if(s.cellParticleCount[c]!=int(expected.size()))return false;
        for(int k=s.cellParticleOffset[c];k<s.cellParticleOffset[c+1];++k)actual.push_back(s.sortedParticleIndex[k]);
        std::sort(actual.begin(),actual.end());if(actual!=expected)return false;
        expectedTotal+=int(expected.size());
    }
    return total==expectedTotal;
}
int main() {
    DeviceState s;checked(cudaMalloc(&s.deviceState,sizeof(s)));
    allocateInt(s.cellParticleCount,9);allocateInt(s.cellParticleOffset,9);allocateInt(s.cellParticleWrite,9);
    allocateInt(s.particleCountDevice,1);allocateInt(s.pCellId,s.particleCapacity);allocateInt(s.pStatus,s.particleCapacity);allocateInt(s.sortedParticleIndex,s.particleCapacity);
    checked(cub::DeviceScan::ExclusiveSum(nullptr,s.cellScanTempBytes,s.cellParticleCount,s.cellParticleOffset,9));
    checked(cudaMalloc(&s.cellScanTempStorage,s.cellScanTempBytes));int* total=nullptr;allocateInt(total,1);
    int captures[3]={},runs=0;
    auto run=[&](int kind,int seed,int count,bool expectedCapture) {
        publish(s);producer<<<9,32>>>(s.deviceState,seed,count);
        const int before=captures[kind];
        if(submitParticleDirectoryGraph(&s,kind,32,[&](cudaStream_t stream){++captures[kind];return submitFull(&s,kind==1,stream);}))std::exit(3);
        consumer<<<1,1>>>(s.deviceState,total);checked(cudaDeviceSynchronize());
        if(captures[kind]-before!=int(expectedCapture)||!verify(s,seed,count,kind==1,*total)){
            std::fprintf(stderr,"FAIL graph kind=%d seed=%d count=%d capture=%d\n",kind,seed,count,captures[kind]-before);std::exit(4);
        }
        ++runs;
    };
    for(int k=0;k<2;++k){run(k,1,257,true);run(k,2,103,false);run(k,3,0,false);}
    s.particleWorkGrid=5;run(0,4,251,true);run(0,5,193,false);
    s.particleBlockThreads=64;run(0,6,257,true);
    s.csrWarpAggregatedBinning=1;run(0,7,257,true);run(0,8,199,false);
    s.nBoundarySources=1;run(0,9,257,true);
    s.nCells=8;run(0,10,257,true);
    int* old=s.cellParticleCount;allocateInt(s.cellParticleCount,9);run(0,11,257,true);checked(cudaFree(old));
    old=s.cellParticleOffset;allocateInt(s.cellParticleOffset,9);run(0,12,257,true);checked(cudaFree(old));
    void* oldTemporary=s.cellScanTempStorage;checked(cudaMalloc(&s.cellScanTempStorage,s.cellScanTempBytes));run(0,13,257,true);checked(cudaFree(oldTemporary));
    auto* oldState=s.deviceState;checked(cudaMalloc(&s.deviceState,sizeof(s)));run(0,14,257,true);run(1,14,257,true);checked(cudaFree(oldState));
    ugkp::gpu::releaseParticleDirectoryGraphs(s.particleDirectoryGraphs);run(0,15,257,true);
    ugkp::gpu::releaseParticleDirectoryGraphs(s.particleDirectoryGraphs);
    checked(cudaFree(s.deviceState));checked(cudaFree(s.cellScanTempStorage));checked(cudaFree(s.cellParticleCount));checked(cudaFree(s.cellParticleOffset));checked(cudaFree(s.cellParticleWrite));checked(cudaFree(s.particleCountDevice));checked(cudaFree(s.pCellId));checked(cudaFree(s.pStatus));checked(cudaFree(s.sortedParticleIndex));checked(cudaFree(total));
    std::printf("PASS directory graph production-kernel runs=%d captures=%d/%d default-stream-ordering replay invalidation empty survivor warp\n",runs,captures[0],captures[1]);
}
