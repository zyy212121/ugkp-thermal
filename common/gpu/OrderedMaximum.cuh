#ifndef UGKP_GPU_ORDERED_MAXIMUM_CUH
#define UGKP_GPU_ORDERED_MAXIMUM_CUH
#include <cuda_runtime.h>
#include <cmath>
namespace ugkp { namespace gpu {
// A segment of the native ordered max(a,b) = a>b ? a : b fold. NaN
// discards the preceding prefix; retain the suffix after its last occurrence.
// This associative segment representation permits parallel reduction without
// changing the original non-finite semantics or treating ignored faces as zero.
struct OrderedMaximum {double value;int reset;int valid;};
__host__ __device__ inline OrderedMaximum maximumSegment(double value){return {value,std::isnan(value),1};}
__host__ __device__ inline OrderedMaximum joinMaximum(OrderedMaximum left,OrderedMaximum right){
    if(!right.valid)return left;if(!left.valid||right.reset)return right;
    return {left.value>right.value?left.value:right.value,left.reset,1};
}
__host__ __device__ inline double applyMaximum(double initial,OrderedMaximum segment){
    if(!segment.valid)return initial;
    return segment.reset?segment.value:(initial>segment.value?initial:segment.value);
}
__global__ void orderedMaximumPartials(int count,const double* input,const unsigned char* valid,OrderedMaximum* output){
    __shared__ OrderedMaximum values[256];
    const int i=blockIdx.x*256+threadIdx.x,batch=blockIdx.y;
    const std::size_t index=std::size_t(batch)*count+i;
    values[threadIdx.x]=i<count&&(!valid||valid[index])?maximumSegment(input[index]):OrderedMaximum{0,0,0};
    __syncthreads();
    for(int stride=1;stride<256;stride*=2){
        if(threadIdx.x%(2*stride)==0)values[threadIdx.x]=joinMaximum(values[threadIdx.x],values[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)output[batch*gridDim.x+blockIdx.x]=values[0];
}
__global__ void orderedMaximumMerge(int count,const OrderedMaximum* input,OrderedMaximum* output){
    __shared__ OrderedMaximum values[256];
    const int i=blockIdx.x*256+threadIdx.x,batch=blockIdx.y;
    values[threadIdx.x]=i<count?input[std::size_t(batch)*count+i]:OrderedMaximum{0,0,0};
    __syncthreads();
    for(int stride=1;stride<256;stride*=2){
        if(threadIdx.x%(2*stride)==0)values[threadIdx.x]=joinMaximum(values[threadIdx.x],values[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x)output[batch*gridDim.x+blockIdx.x]=values[0];
}
}}
#endif
