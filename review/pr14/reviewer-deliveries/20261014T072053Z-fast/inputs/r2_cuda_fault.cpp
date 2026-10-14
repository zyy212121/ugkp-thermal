#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdint>
#include <limits>
#include <cstdio>
static int nthMalloc=0,mallocCount=0,copyCount=0,freeCount=0,observedError=0;
static std::size_t copyBytes=0;
extern "C" void pr14_fault_malloc(int n){nthMalloc=n;mallocCount=copyCount=freeCount=observedError=0;copyBytes=0;}
extern "C" void pr14_fault_copy(std::size_t n){copyBytes=n;nthMalloc=0;mallocCount=copyCount=freeCount=observedError=0;}
extern "C" int pr14_observed_error(){return observedError;}
extern "C" int pr14_malloc_count(){return mallocCount;}
extern "C" int pr14_copy_count(){return copyCount;}
extern "C" int pr14_free_count(){return freeCount;}
extern "C" cudaError_t cudaMalloc(void** out,std::size_t bytes){
    using F=cudaError_t(*)(void**,std::size_t);static auto real=reinterpret_cast<F>(dlsym(RTLD_NEXT,"cudaMalloc"));
    ++mallocCount;const bool selected=nthMalloc>0&&mallocCount==nthMalloc;
    const auto actual=real(out,selected?std::size_t(std::numeric_limits<std::ptrdiff_t>::max()):bytes);
    if(selected){observedError=int(actual);std::fprintf(stderr,"REAL_CUDA_ALLOC_ERROR code=%d requested_normal_bytes=%zu\n",int(actual),bytes);nthMalloc=0;}
    return actual;
}
extern "C" cudaError_t cudaMemcpy(void* out,const void* in,std::size_t bytes,cudaMemcpyKind kind){
    using F=cudaError_t(*)(void*,const void*,std::size_t,cudaMemcpyKind);static auto real=reinterpret_cast<F>(dlsym(RTLD_NEXT,"cudaMemcpy"));
    ++copyCount;const bool selected=copyBytes>0&&bytes==copyBytes&&kind==cudaMemcpyHostToDevice;
    const auto actual=real(out,in,bytes,selected?static_cast<cudaMemcpyKind>(42):kind);
    if(selected){observedError=int(actual);std::fprintf(stderr,"REAL_CUDA_COPY_ERROR code=%d bytes=%zu\n",int(actual),bytes);copyBytes=0;}
    return actual;
}
extern "C" cudaError_t cudaFree(void* pointer){
    using F=cudaError_t(*)(void*);static auto real=reinterpret_cast<F>(dlsym(RTLD_NEXT,"cudaFree"));++freeCount;return real(pointer);
}
