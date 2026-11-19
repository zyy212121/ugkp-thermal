#ifndef UGKP_GPU_ORDERED_MAXIMUM_CUH
#define UGKP_GPU_ORDERED_MAXIMUM_CUH
#include <cuda_runtime.h>
#include <cmath>
namespace ugkp { namespace gpu {
template<class Real>
struct OrderedMaximumSegment {using value_type=Real;Real value;int reset;int valid;};
using OrderedMaximum=OrderedMaximumSegment<double>;
template<class Real> __host__ __device__ inline OrderedMaximumSegment<Real> maximumSegment(Real value){
    return {value,std::isnan(value),1};
}
template<class Real> __host__ __device__ inline OrderedMaximumSegment<Real> joinMaximum(
    OrderedMaximumSegment<Real> left,OrderedMaximumSegment<Real> right){
    if(!right.valid)return left;
    if(!left.valid||right.reset)return right;
    return {left.value>right.value?left.value:right.value,left.reset,1};
}
template<class Initial,class Real> __host__ __device__ inline Real applyMaximum(Initial initialValue,OrderedMaximumSegment<Real> segment){
    const Real initial=Real(initialValue);
    if(!segment.valid)return initial;
    return segment.reset?segment.value:(initial>segment.value?initial:segment.value);
}
template<class T> __device__ inline void reduceOrderedMaximumBlock(T* values){
    __syncthreads();
    for(int stride=1;stride<blockDim.x;stride*=2){
        if(threadIdx.x%(2*stride)==0&&threadIdx.x+stride<blockDim.x)
            values[threadIdx.x]=joinMaximum(values[threadIdx.x],values[threadIdx.x+stride]);
        __syncthreads();
    }
}
template<bool Absolute=false,class Real>
static __global__ void orderedMaximumPartials(int count,const Real* input,const unsigned char* valid,
    OrderedMaximumSegment<Real>* partial,Real* result=nullptr,Real initial=0){
    extern __shared__ __align__(16) unsigned char reductionStorage[];
    auto* values=reinterpret_cast<OrderedMaximumSegment<Real>*>(reductionStorage);
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto index=std::size_t(blockIdx.y)*count+i;
    OrderedMaximumSegment<Real> entry{};
    if(i<count&&(!valid||valid[index])){
        const Real v=input[index];entry=maximumSegment(Absolute?Real(fabs(v)):v);
    }
    values[threadIdx.x]=entry;reduceOrderedMaximumBlock(values);
    if(!threadIdx.x){
        if(result)result[blockIdx.y]=applyMaximum(initial,values[0]);
        else partial[std::size_t(blockIdx.y)*gridDim.x+blockIdx.x]=values[0];
    }
}
template<class Real>
static __global__ void orderedMaximumMerge(int count,const OrderedMaximumSegment<Real>* input,OrderedMaximumSegment<Real>* partial,
    Real* result=nullptr,Real initial=0){
    extern __shared__ __align__(16) unsigned char reductionStorage[];
    auto* values=reinterpret_cast<OrderedMaximumSegment<Real>*>(reductionStorage);
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    values[threadIdx.x]=i<count?input[std::size_t(blockIdx.y)*count+i]:OrderedMaximumSegment<Real>{};
    reduceOrderedMaximumBlock(values);
    if(!threadIdx.x){
        if(result)result[blockIdx.y]=applyMaximum(initial,values[0]);
        else partial[std::size_t(blockIdx.y)*gridDim.x+blockIdx.x]=values[0];
    }
}
template<bool Absolute=false,class Real>
inline void enqueueOrderedMaximum(int count,const Real* input,const unsigned char* valid,
    typename OrderedMaximumSegment<Real>::value_type initial,Real* result,OrderedMaximumSegment<Real>* current,OrderedMaximumSegment<Real>* scratch,
    int width,cudaStream_t stream){
    int blocks=count?(count-1)/width+1:1;
    orderedMaximumPartials<Absolute><<<blocks,width,width*sizeof(OrderedMaximumSegment<Real>),stream>>>(
        count,input,valid,current,blocks==1?result:nullptr,initial);
    while(blocks>1){
        const int next=(blocks-1)/width+1;
        orderedMaximumMerge<<<next,width,width*sizeof(OrderedMaximumSegment<Real>),stream>>>(blocks,current,scratch,next==1?result:nullptr,initial);
        auto* saved=current;current=scratch;scratch=saved;blocks=next;
    }
}
}}
#endif
