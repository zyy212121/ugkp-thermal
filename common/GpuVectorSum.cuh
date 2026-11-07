#pragma once
#include <cuda_runtime.h>

namespace ugkwp {
// Fixed-size FP64 inventory reductions. Sources provide count() and add(i,q).
// The workspace and both launches stay on the caller's stream; the final sum
// has one writer per component. No atomics or host convergence/readback.
template<int Components,int Block,class Source>
__global__ void vectorSumPartials(Source source,double* partials)
{
    static_assert(Block%32==0 && Block<=1024,"whole CUDA warps required");
    __shared__ double warpSums[Components][Block/32];
    double q[Components]={};
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int count=source.count();
    for(size_t i=size_t(blockIdx.x)*Block+threadIdx.x;i<size_t(count);
        i+=size_t(gridDim.x)*Block)source.add(int(i),q);
    for(int k=0;k<Components;++k){
        for(int d=16;d;d/=2)q[k]+=__shfl_down_sync(0xffffffffu,q[k],d);
        if(!lane)warpSums[k][warp]=q[k];
    }
    __syncthreads();
    if(!warp)for(int k=0;k<Components;++k){
        double value=lane<Block/32?warpSums[k][lane]:0.;
        for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffffu,value,d);
        if(!lane)partials[size_t(k)*gridDim.x+blockIdx.x]=value;
    }
}
template<int Block>
__global__ void addVectorSumPartials(const double* partials,int parts,double* output,double scale)
{
    __shared__ double sums[Block];
    const int t=threadIdx.x,k=blockIdx.x;
    double value=0.;
    for(int i=t;i<parts;i+=Block)value+=partials[size_t(k)*parts+i];
    sums[t]=value;__syncthreads();
    for(int n=Block/2;n;n/=2){if(t<n)sums[t]+=sums[t+n];__syncthreads();}
    if(!t)output[k]+=scale*sums[0];
}
template<int Components,class Source>
inline cudaError_t accumulateVectorSum(Source source,int parts,double* partials,
    double* output,double scale,cudaStream_t stream)
{
    vectorSumPartials<Components,256><<<parts,256,0,stream>>>(source,partials);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    addVectorSumPartials<256><<<Components,256,0,stream>>>(partials,parts,output,scale);
    return cudaGetLastError();
}
} // namespace ugkwp
