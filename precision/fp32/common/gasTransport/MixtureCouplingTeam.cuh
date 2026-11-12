#pragma once
#include "../GpuPrecisionTypes.H"
// Shared execution policy for finite-bath drag and caloric heat transactions.
namespace mixtureParticles {
struct SerialCouplingTeam {
 __device__ int rank()const{return 0;} __device__ int width()const{return 1;}
 __device__ GpuReal sum(GpuReal x)const{return x;}
 __device__ GpuReal minimum(GpuReal x)const{return x;}
 __device__ GpuReal maximum(GpuReal x)const{return x;}
 template<class T> __device__ T broadcast(T x)const{return x;}
 __device__ void sync()const{}
};
#if defined(__CUDACC__)
struct WarpCouplingTeam {
 __device__ int rank()const{return threadIdx.x%32;} __device__ int width()const{return 32;}
 __device__ GpuReal sum(GpuReal x)const{
  for(int offset=16;offset;offset/=2)x+=__shfl_down_sync(0xffffffffu,x,offset);
  return __shfl_sync(0xffffffffu,x,0);
 }
 __device__ GpuReal minimum(GpuReal x)const{
  for(int offset=16;offset;offset/=2)x=fmin(x,__shfl_down_sync(0xffffffffu,x,offset));
  return __shfl_sync(0xffffffffu,x,0);
 }
 __device__ GpuReal maximum(GpuReal x)const{
  for(int offset=16;offset;offset/=2)x=fmax(x,__shfl_down_sync(0xffffffffu,x,offset));
  return __shfl_sync(0xffffffffu,x,0);
 }
 template<class T> __device__ T broadcast(T x)const{return __shfl_sync(0xffffffffu,x,0);}
 __device__ void sync()const{__syncwarp();}
};
#endif
}
