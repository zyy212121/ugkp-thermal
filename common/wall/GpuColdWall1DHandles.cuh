#pragma once
#include <cuda_runtime.h>
#include <type_traits>
namespace Foam { namespace gpuWall {
template<class State, class=void> struct HasColdHandles : std::false_type {};
template<class State> struct HasColdHandles<State,std::void_t<decltype(State::pColdHandle)>> : std::true_type {};
template<class State>
__host__ __device__ inline int coldWallSlot(const State& s,int i,bool compact=false) {
    if constexpr(HasColdHandles<State>::value)
        return compact?s.compactPColdHandle[i]:s.pColdHandle[i];
    else return i;
}
template<class State,class Scalar>
__device__ inline int coldWallFieldSlot(const State& s,const Scalar* field,int i) {
    const void* p=field;
    return p==static_cast<const void*>(s.pColdNodeSpecificEnthalpy)
        || p==static_cast<const void*>(s.pColdRingSolidMass)
        || p==static_cast<const void*>(s.pColdFrozenArea)
        || p==static_cast<const void*>(s.pColdContactAge)
        ? coldWallSlot(s,i) : i;
}



struct ColdWall1DHandles {
    int* particle; int* compact; int* freeHandles; int* counts; int capacity;
};
static __global__ void initialiseColdWall1DHandlesKernel(ColdWall1DHandles p,int restoredCount=0) {
    const int n=restoredCount<0?0:(restoredCount>p.capacity?p.capacity:restoredCount);
    if(blockIdx.x==0 && threadIdx.x==0) {
        p.counts[0]=p.capacity-n;
    }
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<p.capacity;i+=blockDim.x*gridDim.x) {
        p.particle[i]=i<n?i:-1; p.compact[i]=-1;

        if(i<p.capacity-n)p.freeHandles[i]=p.capacity-1-i;
    }
}
template<class State>
__device__ inline void releaseColdWall1DHandle(const State& s,int i) {
    if constexpr(HasColdHandles<State>::value) {
        if(!s.pColdHandle)return;
        const int handle=atomicExch(s.pColdHandle+i,-1);
        if(handle>=0) {
            const int freeIndex=atomicAdd(s.coldFreeCounts,1);
            s.coldFreeHandles[freeIndex]=handle;
        }
    }
}
template<class State>
__device__ inline void allocateColdWall1DHandle(const State& s,int i) {
    if constexpr(HasColdHandles<State>::value) {
        if(!s.pColdHandle)return;


        const int freeIndex=atomicSub(s.coldFreeCounts,1)-1;
        s.pColdHandle[i]=s.coldFreeHandles[freeIndex];
    }
}
template<class Scalar,int Width,class State>
__global__ void gatherColdWall1DStateKernel
(const State* sp,const Scalar* pool,Scalar* packed,const int count) {
    const int* handles=sp->pColdHandle;
    for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<count*Width;j+=blockDim.x*gridDim.x)
        packed[j]=handles[j/Width]>=0?pool[handles[j/Width]*Width+j%Width]:Scalar(0);
}
} }
