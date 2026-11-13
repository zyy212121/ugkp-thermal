#pragma once
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <limits>
#include <cmath>
namespace ugkwp {
struct ScalarGasCourantMaximum {
    __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
template<class Real>
inline double reduceGasAcousticCourant(const Real* acoustic,int count,cudaStream_t stream){
    return thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(acoustic),
        thrust::device_pointer_cast(acoustic)+count,-std::numeric_limits<double>::infinity(),
        ScalarGasCourantMaximum{});
}
} // namespace ugkwp
