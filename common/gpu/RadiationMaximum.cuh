#ifndef UGKP_GPU_RADIATION_MAXIMUM_CUH
#define UGKP_GPU_RADIATION_MAXIMUM_CUH
#include "RadiationPrecision.H"
#include <cuda_runtime.h>
#include <cmath>
namespace ugkp { namespace gpu {




struct RadiationMaximum {RadiationReal value;int reset;int valid;};
__host__ __device__ inline RadiationMaximum radiationMaximumSegment(RadiationReal value){return {value,std::isnan(value),1};}
__host__ __device__ inline RadiationMaximum joinRadiationMaximum(RadiationMaximum left,RadiationMaximum right){
    if(!right.valid)return left;if(!left.valid||right.reset)return right;
    return {left.value>right.value?left.value:right.value,left.reset,1};
}
__host__ __device__ inline RadiationReal applyRadiationMaximum(RadiationReal initial,RadiationMaximum segment){
    if(!segment.valid)return initial;
    return segment.reset?segment.value:(initial>segment.value?initial:segment.value);
}
static __global__ void radiationMaximumPartials(int count,const RadiationReal* input,const unsigned char* valid,RadiationMaximum* output){
    extern __shared__ RadiationMaximum values[];
    const int i=blockIdx.x*blockDim.x+threadIdx.x,batch=blockIdx.y;
    const std::size_t index=std::size_t(batch)*count+i;
    values[threadIdx.x]=i<count&&(!valid||valid[index])?radiationMaximumSegment(input[index]):RadiationMaximum{0,0,0};
    __syncthreads();
    for(int stride=1;stride<blockDim.x;stride*=2){
        if(threadIdx.x%(2*stride)==0 && threadIdx.x+stride<blockDim.x)values[threadIdx.x]=joinRadiationMaximum(values[threadIdx.x],values[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)output[batch*gridDim.x+blockIdx.x]=values[0];
}
static __global__ void radiationMaximumMerge(int count,const RadiationMaximum* input,RadiationMaximum* output){
    extern __shared__ RadiationMaximum values[];
    const int i=blockIdx.x*blockDim.x+threadIdx.x,batch=blockIdx.y;
    values[threadIdx.x]=i<count?input[std::size_t(batch)*count+i]:RadiationMaximum{0,0,0};
    __syncthreads();
    for(int stride=1;stride<blockDim.x;stride*=2){
        if(threadIdx.x%(2*stride)==0 && threadIdx.x+stride<blockDim.x)values[threadIdx.x]=joinRadiationMaximum(values[threadIdx.x],values[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)output[batch*gridDim.x+blockIdx.x]=values[0];
}
}}
#endif
