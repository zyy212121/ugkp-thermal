#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <new>
#include "OrderedBisectionProbe.H"
namespace ugkp { namespace gpu {
// Each subteam owns an independent evaluator workspace. The complete block
// follows one root trajectory; unconsumed probes never publish physical state.
template<int Depth,class Subteam> struct OrderedBisectionTeam {
    static constexpr int probeDepth=Depth,probeCount=(1<<Depth)-1,width=probeCount*Subteam::width;
    template<class Value> struct Storage {
        Value result[probeCount];bool valid[probeCount],numerical[probeCount];std::size_t workspaceStride;
    };
    template<class Value> __device__ static Storage<Value>& storage(){
        static_assert(alignof(Storage<Value>)<=16,"probe scratch alignment");
        extern __shared__ __align__(16) unsigned char orderedBisectionScratch[];
        return *reinterpret_cast<Storage<Value>*>(orderedBisectionScratch);
    }
    template<class Value> __device__ void initialize(std::size_t stride)const{
        if(leader()){auto& slots=storage<Value>();new(&slots)Storage<Value>{};slots.workspaceStride=stride;}
        sync();
    }
    __device__ int probe()const{return int(threadIdx.x)/Subteam::width;}
    __device__ Subteam inner()const{return {};}
    __device__ bool leader()const{return threadIdx.x==0;}
    __device__ void sync()const{__syncthreads();}
    __device__ bool broadcast(bool value)const{return __syncthreads_or(leader()&&value)!=0;}
};
// Scheduling uses the compiled kernel's actual residency, queried at setup.
// It adds no mesh, material, tolerance, or user-facing scheduling parameter.
template<class Kernel> inline cudaError_t residentProbeBlocks(Kernel kernel,int threads,
    std::size_t sharedBytes,int multiprocessors,int& total){
    int blocks=0;const auto error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,sharedBytes);
    if(error!=cudaSuccess)return error;
    total=blocks*multiprocessors;return cudaSuccess;
}
}}
