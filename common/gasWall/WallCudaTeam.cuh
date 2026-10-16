#pragma once
#include "WallExecution.H"
namespace ugkwp { namespace gaswall { namespace detail {
// One full hardware warp owns one wall workspace. All lanes participate in
// every collective; the sole launch site and launch bound share this width.
struct CudaWallTeam {
    static constexpr int width=wallWarpWidth;
    static constexpr unsigned mask=0xffffffffu;
    __device__ int rank()const{return int(threadIdx.x);}
    __device__ int size()const{return width;}
    __device__ bool leader()const{return threadIdx.x==0;}
    __device__ void sync()const{__syncwarp(mask);}
    __device__ bool all(bool value)const{
        // Vote/shuffle alone do not order workspace memory. The explicit warp
        // barrier publishes all preceding writes before a consumer proceeds.
        sync();return __all_sync(mask,int(value))!=0;
    }
    __device__ bool broadcast(bool value)const{
        sync();return __shfl_sync(mask,int(value),0)!=0;
    }
    __device__ double broadcast(double value)const{
        sync();return __shfl_sync(mask,value,0);
    }
};
}}}
