#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

namespace ugkwp {
// A maximum has no rounding-sensitive sum order. Propagate nonfinite input
// instead of letting an unavailable equation look converged.
__device__ inline double componentMaximum(double a,double b)
{
    return isnan(a)?a:isnan(b)?b:fmax(a,b);
}
template<int Block>
__global__ void componentMaximumPartials(const double* values,int count,int parts,double* output)
{
    __shared__ double shared[Block];
    const int lane=threadIdx.x,component=blockIdx.y;
    double value=-INFINITY;
    for(size_t i=size_t(blockIdx.x)*Block+lane;i<size_t(count);i+=size_t(parts)*Block)
        value=componentMaximum(value,values[size_t(component)*count+i]);
    shared[lane]=value;__syncthreads();
    for(int n=Block/2;n;n/=2){
        if(lane<n)shared[lane]=componentMaximum(shared[lane],shared[lane+n]);
        __syncthreads();
    }
    if(!lane)output[size_t(component)*parts+blockIdx.x]=shared[0];
}
inline int componentMaximumParts(int count){return count>0?1+(count-1)/256:1;}
inline cudaError_t batchedComponentMaximum(const double* values,int count,int components,
    double* workspace,cudaStream_t stream)
{
    const int parts=componentMaximumParts(count);
    componentMaximumPartials<256><<<dim3(parts,components),256,0,stream>>>(values,count,parts,workspace);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    componentMaximumPartials<256><<<dim3(1,components),256,0,stream>>>
        (workspace,parts,1,workspace+size_t(parts)*components);
    return cudaGetLastError();
}
// Component-major statistics: maxima first, then sums. All components share
// the same reduction schedule and publish one compact device result array.
template<int Block>
__global__ void componentStatisticsPartials(const double* values,int count,int parts,
    int maximumComponents,double* output)
{
    __shared__ double shared[Block];
    const int lane=threadIdx.x,component=blockIdx.y;
    const bool maximum=component<maximumComponents;
    double value=maximum?-INFINITY:0.0;
    for(size_t i=size_t(blockIdx.x)*Block+lane;i<size_t(count);i+=size_t(parts)*Block)
        value=maximum?componentMaximum(value,values[size_t(component)*count+i])
            :value+values[size_t(component)*count+i];
    shared[lane]=value;__syncthreads();
    for(int n=Block/2;n;n/=2){
        if(lane<n)shared[lane]=maximum?componentMaximum(shared[lane],shared[lane+n])
            :shared[lane]+shared[lane+n];
        __syncthreads();
    }
    if(!lane)output[size_t(component)*parts+blockIdx.x]=shared[0];
}
inline cudaError_t batchedComponentStatistics(const double* values,int count,int components,
    int maximumComponents,double* workspace,cudaStream_t stream)
{
    const int parts=componentMaximumParts(count);
    componentStatisticsPartials<256><<<dim3(parts,components),256,0,stream>>>
        (values,count,parts,maximumComponents,workspace);
    auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    componentStatisticsPartials<256><<<dim3(1,components),256,0,stream>>>
        (workspace,parts,1,maximumComponents,workspace+size_t(parts)*components);
    return cudaGetLastError();
}
} // namespace ugkwp
