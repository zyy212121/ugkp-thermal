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
    __device__ double sum(double value)const{
        for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(mask,value,offset);
        return __shfl_sync(mask,value,0);
    }
    __device__ double maximum(double value)const{
        for(int offset=16;offset;offset/=2){double other=__shfl_down_sync(mask,value,offset);value=value>other?value:other;}
        return __shfl_sync(mask,value,0);
    }
    __device__ double broadcast(double value)const{
        sync();return __shfl_sync(mask,value,0);
    }
};
// Four independent eight-lane teams may share one hardware warp. Each
// collective names only its own tile, including when another wall has finished.
struct CudaWallTile8 {
    static constexpr int width=8;
    __device__ int rank()const{return int(threadIdx.x)&7;}
    __device__ int size()const{return width;}
    __device__ bool leader()const{return rank()==0;}
    __device__ unsigned mask()const{return 0xffu<<(int(threadIdx.x)&24);}
    __device__ void sync()const{__syncwarp(mask());}
    __device__ bool all(bool value)const{sync();return __all_sync(mask(),int(value))!=0;}
    __device__ bool broadcast(bool value)const{sync();return __shfl_sync(mask(),int(value),0,width)!=0;}
    __device__ double broadcast(double value)const{sync();return __shfl_sync(mask(),value,0,width);}
    __device__ double sum(double value)const{
        for(int offset=4;offset;offset/=2)value+=__shfl_down_sync(mask(),value,offset,width);
        return __shfl_sync(mask(),value,0,width);
    }
    __device__ double maximum(double value)const{
        for(int offset=4;offset;offset/=2){double other=__shfl_down_sync(mask(),value,offset,width);value=value>other?value:other;}
        return __shfl_sync(mask(),value,0,width);
    }
};
}}}
