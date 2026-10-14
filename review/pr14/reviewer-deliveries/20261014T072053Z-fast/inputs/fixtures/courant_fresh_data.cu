
#include "gpu/CourantReductionStorage.cuh"
#include "gasTransport/GasCapabilities.H"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
using chmt::CoupledCourantStorage;
struct CoupledCourantMaximum {
    __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
struct CoupledCourantValue {double value;bool unavailable;unsigned char initializedTail[7]{};};
struct CoupledCourantRead {
    const double* field;const int* status;
    __host__ __device__ CoupledCourantValue operator()(int cell)const{
        return {field[cell],status&&status[cell]==int(ugkwp::GasTransportCode::InvalidStorage)};
    }
};
struct CoupledCourantReduce {
    __host__ __device__ CoupledCourantValue operator()(CoupledCourantValue a,CoupledCourantValue b)const{
        return {CoupledCourantMaximum{}(a.value,b.value),a.unavailable||b.unavailable};
    }
};

static_assert(sizeof(CoupledCourantValue)==16,"Preserve reduction value size");
static_assert(offsetof(CoupledCourantValue,unavailable)==8,"Preserve bool offset");
static_assert(offsetof(CoupledCourantValue,initializedTail)==9,"Cover all tail bytes");
static_assert(std::is_same<decltype(CoupledCourantValue::value),double>::value,"FP64 value");
static_assert(std::is_same<decltype(CoupledCourantValue::unavailable),bool>::value,"Bool status");
static_assert(!std::is_copy_constructible<CoupledCourantStorage>::value,"Unique owner");
static_assert(!std::is_move_constructible<CoupledCourantStorage>::value,"Stable fault owner");
void check(cudaError_t error){if(error!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(error));std::exit(2);}}
void sameBits(double a,double b){assert(std::memcmp(&a,&b,sizeof(double))==0);}
void runCase(int count,int variant,cudaStream_t stream){
    std::vector<double> input(count);
    std::vector<int> status(count);
    for(int i=0;i<count;++i){input[i]=double(i%53)-17.;status[i]=i%17==3?int(ugkwp::GasTransportCode::InvalidStorage):0;}
    if(variant==1){for(int i=0;i<count;++i)input[i]=i%2?-0.0:0.0;}
    if(variant==2 && count){
        const std::uint64_t first=0x7ff8000000000042ULL,second=0x7ff8000000000137ULL;
        std::memcpy(&input.front(),&first,sizeof(double));
        if(count>1)std::memcpy(&input.back(),&second,sizeof(double));
    }
    double* field=nullptr;int* deviceStatus=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&field),std::max(count,1)*sizeof(double)));
    check(cudaMalloc(reinterpret_cast<void**>(&deviceStatus),std::max(count,1)*sizeof(int)));
    if(count){check(cudaMemcpyAsync(field,input.data(),count*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(deviceStatus,status.data(),count*sizeof(int),cudaMemcpyHostToDevice,stream));}
    CoupledCourantStorage storage;
    const auto first=thrust::make_counting_iterator(0);
    const CoupledCourantValue initial{-std::numeric_limits<double>::infinity(),false};
    const auto expected=thrust::transform_reduce(thrust::cuda::par.on(stream),first,first+count,
        CoupledCourantRead{field,deviceStatus},initial,CoupledCourantReduce{});
    const auto scalar=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(field),
        thrust::device_pointer_cast(field)+count,initial.value,CoupledCourantMaximum{});
    for(int repeat=0;repeat<100;++repeat){
        const auto value=thrust::transform_reduce(thrust::cuda::par(storage).on(stream),first,first+count,
            CoupledCourantRead{field,deviceStatus},initial,CoupledCourantReduce{});
        sameBits(value.value,expected.value);assert(value.unavailable==expected.unavailable);
        for(auto byte:value.initializedTail)assert(byte==0);
        const auto maximum=thrust::reduce(thrust::cuda::par(storage).on(stream),thrust::device_pointer_cast(field),
            thrust::device_pointer_cast(field)+count,initial.value,CoupledCourantMaximum{});
        sameBits(maximum,scalar);
    }
    check(cudaFree(deviceStatus));check(cudaFree(field));
}

void freshData(CoupledCourantStorage& storage,int count,int round,cudaStream_t stream){
 std::vector<double> h(count);std::vector<int> st(count);
 for(int i=0;i<count;++i){h[i]=double((i*(round+3)+round*13)%109)-77.-double(round)/8;st[i]=(i+round)%19==0?int(ugkwp::GasTransportCode::InvalidStorage):0;}
 if(count && round%7==0){std::uint64_t nan=0x7ff8000000000000ULL+round;std::memcpy(&h[count/2],&nan,8);}
 double*f=nullptr;int*s=nullptr;check(cudaMalloc(&f,std::max(count,1)*sizeof(double)));check(cudaMalloc(&s,std::max(count,1)*sizeof(int)));
 if(count){check(cudaMemcpyAsync(f,h.data(),count*8,cudaMemcpyHostToDevice,stream));check(cudaMemcpyAsync(s,st.data(),count*4,cudaMemcpyHostToDevice,stream));}
 auto first=thrust::make_counting_iterator(0);CoupledCourantValue init{-std::numeric_limits<double>::infinity(),false};
 auto expected=thrust::transform_reduce(thrust::cuda::par.on(stream),first,first+count,CoupledCourantRead{f,s},init,CoupledCourantReduce{});
 auto actual=thrust::transform_reduce(thrust::cuda::par(storage).on(stream),first,first+count,CoupledCourantRead{f,s},init,CoupledCourantReduce{});
 sameBits(actual.value,expected.value);assert(actual.unavailable==expected.unavailable);for(auto b:actual.initializedTail)assert(b==0);
 auto expectedScalar=thrust::reduce(thrust::cuda::par.on(stream),thrust::device_pointer_cast(f),thrust::device_pointer_cast(f)+count,init.value,CoupledCourantMaximum{});
 auto scalar=thrust::reduce(thrust::cuda::par(storage).on(stream),thrust::device_pointer_cast(f),thrust::device_pointer_cast(f)+count,init.value,CoupledCourantMaximum{});sameBits(scalar,expectedScalar);
 check(cudaFree(s));check(cudaFree(f));
 std::printf("FRESH_DATA n=%d round=%d result=%a unavailable=%d\n",count,round,actual.value,int(actual.unavailable));
}
int main(){
    int devices=0;check(cudaGetDeviceCount(&devices));assert(devices>0);
    {
        CoupledCourantStorage storage;
        char* first=storage.allocate(17);storage.deallocate(first,17);
        char* smaller=storage.allocate(9);assert(first==smaller);
        char* simultaneous=storage.allocate(17);assert(smaller!=simultaneous);
        storage.deallocate(smaller,9);storage.deallocate(simultaneous,17);
        char* grown=storage.allocate(4096);check(cudaMemset(grown,0x5a,4096));storage.deallocate(grown,4096);
        char* retained=storage.allocate(4096);assert(retained==grown);storage.deallocate(retained,4096);
        bool failed=false;try{storage.allocate(std::numeric_limits<std::ptrdiff_t>::max());}
        catch(const std::exception&){failed=true;}assert(failed);
        char* retry=storage.allocate(17);check(cudaMemset(retry,0,17));storage.deallocate(retry,17);
    }
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(auto selected:{cudaStream_t{},stream})for(int count:{0,1,31,32,33,513,32768})
        for(int variant:{0,1,2})runCase(count,variant,selected);
    { CoupledCourantStorage owner,other; int round=0; for(int n:{1,32,513,32768,513,1,0})for(int repeat=0;repeat<12;++repeat){freshData(owner,n,round++,stream);freshData(other,n,round++,nullptr);} }
    check(cudaStreamDestroy(stream));
    std::puts("CUDA Courant reuse/equivalence PASS");
}
