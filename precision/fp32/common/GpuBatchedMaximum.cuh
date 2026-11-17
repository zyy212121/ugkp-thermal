#pragma once
#include <cuda_runtime.h>
#include "gpu/WorkPlan.H"
#include "GpuPrecisionTypes.H"
#include <cmath>
#include <limits>

namespace ugkwp {


__device__ inline GpuReal componentMaximum(GpuReal a,GpuReal b)
{
    return std::isnan(a)?a:std::isnan(b)?b:std::fmax(a,b);
}
static __global__ void componentMaximumPartials(const GpuReal* values,int count,int parts,GpuReal* output)
{
    extern __shared__ GpuReal shared[];
    const int lane=threadIdx.x,component=blockIdx.y;
    GpuReal value=-INFINITY;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+lane;i<size_t(count);i+=size_t(parts)*blockDim.x)
        value=componentMaximum(value,values[size_t(component)*count+i]);
    shared[lane]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){
        if(lane<active/2)shared[lane]=componentMaximum(shared[lane],shared[lane+(active+1)/2]);
        __syncthreads();
    }
    if(!lane)output[size_t(component)*parts+blockIdx.x]=shared[0];
}
inline int componentMaximumParts(int count){return count>0?1+(count-1)/ugkp::gpu::workPlan().reductionBlockThreads:1;}
inline cudaError_t batchedComponentMaximum(const GpuReal* values,int count,int components,
    GpuReal* workspace,cudaStream_t stream)
{
    const int block=ugkp::gpu::workPlan().reductionBlockThreads;
    const int parts=componentMaximumParts(count);
    componentMaximumPartials<<<dim3(parts,components),block,block*sizeof(GpuReal),stream>>>(values,count,parts,workspace);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    componentMaximumPartials<<<dim3(1,components),block,block*sizeof(GpuReal),stream>>>
        (workspace,parts,1,workspace+size_t(parts)*components);
    return cudaGetLastError();
}


static __global__ void componentStatisticsPartials(const GpuReal* values,int count,int parts,
    int maximumComponents,GpuReal* output)
{
    extern __shared__ GpuReal shared[];
    const int lane=threadIdx.x,component=blockIdx.y;
    const bool maximum=component<maximumComponents;
    GpuReal value=maximum?-INFINITY:GpuReal(0);
    for(size_t i=size_t(blockIdx.x)*blockDim.x+lane;i<size_t(count);i+=size_t(parts)*blockDim.x)
        value=maximum?componentMaximum(value,values[size_t(component)*count+i])
            :value+values[size_t(component)*count+i];
    shared[lane]=value;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){
        if(lane<active/2)shared[lane]=maximum?componentMaximum(shared[lane],shared[lane+(active+1)/2])
            :shared[lane]+shared[lane+(active+1)/2];
        __syncthreads();
    }
    if(!lane)output[size_t(component)*parts+blockIdx.x]=shared[0];
}
inline cudaError_t batchedComponentStatistics(const GpuReal* values,int count,int components,
    int maximumComponents,GpuReal* workspace,cudaStream_t stream)
{
    const int block=ugkp::gpu::workPlan().reductionBlockThreads;
    const int parts=componentMaximumParts(count);
    componentStatisticsPartials<<<dim3(parts,components),block,block*sizeof(GpuReal),stream>>>
        (values,count,parts,maximumComponents,workspace);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    componentStatisticsPartials<<<dim3(1,components),block,block*sizeof(GpuReal),stream>>>
        (workspace,parts,1,maximumComponents,workspace+size_t(parts)*components);
    return cudaGetLastError();
}
}
