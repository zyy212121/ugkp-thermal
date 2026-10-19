#pragma once
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <limits>
#include <cmath>
namespace ugkwp {
struct GasCourantComponents {double convective,diffusion,sst;};
struct ReadGasCourantComponents {
    const double *convective,*diffusion,*sst;
    __device__ GasCourantComponents operator()(int i)const{
        return {convective[i],diffusion[i],sst?sst[i]:0.};
    }
};
struct MaxGasCourantComponents {
    __host__ __device__ static double maximum(double a,double b){return a!=a?a:b!=b?b:fmax(a,b);}
    __host__ __device__ GasCourantComponents operator()(GasCourantComponents a,GasCourantComponents b)const{
        return {maximum(a.convective,b.convective),maximum(a.diffusion,b.diffusion),maximum(a.sst,b.sst)};
    }
};
struct ScalarGasCourantMaximum {
    __host__ __device__ double operator()(double a,double b)const{return MaxGasCourantComponents::maximum(a,b);}
};
inline GasCourantComponents reduceGasCourantComponents(const double* convective,
 const double* diffusion,const double* sst,int count,cudaStream_t stream){
    const double identity=-std::numeric_limits<double>::infinity();
    const auto first=thrust::make_counting_iterator(0);
    auto result=thrust::transform_reduce(thrust::cuda::par.on(stream),first,first+count,
        ReadGasCourantComponents{convective,diffusion,sst},
        GasCourantComponents{identity,identity,sst?identity:0.},MaxGasCourantComponents{});
    // Tuple reduction may visit NaNs in a different tree order. Preserve the
    // scalar reduction's exact payload on this exceptional path; finite queries
    // need only the joint reduction and its single host transfer.
    const double* fields[]={convective,diffusion,sst};
    double* values[]={&result.convective,&result.diffusion,&result.sst};
    for(int j=0;j<3;++j)if(*values[j]!=*values[j])
        *values[j]=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(fields[j]),
            thrust::device_pointer_cast(fields[j])+count,identity,ScalarGasCourantMaximum{});
    return result;
}
} // namespace ugkwp
