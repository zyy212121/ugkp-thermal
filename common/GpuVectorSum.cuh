#pragma once
#include <cuda_runtime.h>
#include "gpu/WorkPlan.H"

namespace ugkwp {
// Fixed-size FP64 inventory reductions. Sources provide count() and add(i,q).
// The workspace and both launches stay on the caller's stream; the final sum
// has one writer per component. No atomics or host convergence/readback.
template<int Components,class Source>
__global__ void vectorSumPartials(Source source,double* partials)
{
    extern __shared__ double vectorWarpSums[];
    double q[Components]={};
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int count=source.count();
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(count);
        i+=size_t(gridDim.x)*blockDim.x)source.add(int(i),q);
    for(int k=0;k<Components;++k){
        for(int d=16;d;d/=2)q[k]+=__shfl_down_sync(0xffffffffu,q[k],d);
        if(!lane)vectorWarpSums[k*(blockDim.x/32)+warp]=q[k];
    }
    __syncthreads();
    if(!warp)for(int k=0;k<Components;++k){
        double value=lane<blockDim.x/32?vectorWarpSums[k*(blockDim.x/32)+lane]:0.;
        for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffffu,value,d);
        if(!lane)partials[size_t(k)*gridDim.x+blockIdx.x]=value;
    }
}
static __global__ void addVectorSumPartials(const double* partials,int parts,double* output,double scale)
{
    extern __shared__ double sums[];
    const int t=threadIdx.x,k=blockIdx.x;
    double value=0.;
    for(int i=t;i<parts;i+=blockDim.x)value+=partials[size_t(k)*parts+i];
    sums[t]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){if(t<active/2)sums[t]+=sums[t+(active+1)/2];__syncthreads();}
    if(!t)output[k]+=scale*sums[0];
}
template<int Components,class Source>
inline cudaError_t accumulateVectorSum(Source source,int parts,double* partials,
    double* output,double scale,cudaStream_t stream)
{
    const int block=ugkp::gpu::workPlan().reductionBlockThreads;
    vectorSumPartials<Components><<<parts,block,Components*(block/32)*sizeof(double),stream>>>(source,partials);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    addVectorSumPartials<<<Components,block,block*sizeof(double),stream>>>(partials,parts,output,scale);
    return cudaGetLastError();
}
} // namespace ugkwp
