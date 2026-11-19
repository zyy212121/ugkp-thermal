#ifndef UGKP_GPU_KERNEL_RESOURCE_LAUNCH_CUH
#define UGKP_GPU_KERNEL_RESOURCE_LAUNCH_CUH
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>
namespace ugkp { namespace gpu {
struct KernelResources {int multiprocessors=0,blocksPerMultiprocessor=0;};
template<class Kernel>
cudaError_t kernelResources(Kernel kernel,int blockThreads,std::size_t dynamicSharedBytes,KernelResources& result){
    int device=0;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    struct Entry {int device,threads;std::size_t shared;Kernel kernel;KernelResources resources;};
    static thread_local std::vector<Entry> cache;
    for(const auto& e:cache)if(e.device==device&&e.threads==blockThreads&&e.shared==dynamicSharedBytes&&e.kernel==kernel){result=e.resources;return cudaSuccess;}
    KernelResources resources;
    error=cudaDeviceGetAttribute(&resources.multiprocessors,cudaDevAttrMultiProcessorCount,device);if(error!=cudaSuccess)return error;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resources.blocksPerMultiprocessor,kernel,blockThreads,dynamicSharedBytes);if(error!=cudaSuccess)return error;
    cache.push_back({device,blockThreads,dynamicSharedBytes,kernel,resources});result=resources;return cudaSuccess;
}
struct KernelGridBuckets {int count=0;int blocks[32]{};};
template<class Kernel>
cudaError_t kernelGridBuckets(Kernel kernel,int blockThreads,std::size_t dynamicSharedBytes,
    long long taskBound,int tasksPerBlock,KernelGridBuckets& result){
    KernelResources resources;auto error=kernelResources(kernel,blockThreads,dynamicSharedBytes,resources);if(error!=cudaSuccess)return error;
    const long long work=std::max(1LL,(taskBound+tasksPerBlock-1)/tasksPerBlock);
    const long long resident=std::max(1LL,2LL*resources.multiprocessors*resources.blocksPerMultiprocessor);
    const int maximum=int(std::min(static_cast<long long>(INT_MAX),std::min(work,resident)));
    result={};int blocks=1;
    while(blocks<maximum){result.blocks[result.count++]=blocks;if(blocks>maximum/2)break;blocks*=2;}
    if(!result.count||result.blocks[result.count-1]!=maximum)result.blocks[result.count++]=maximum;
    return cudaSuccess;
}
static __device__ unsigned int selectedKernelGrid(const int* count,int tasksPerBlock,int extra,KernelGridBuckets buckets){
    const long long tasks=(static_cast<long long>(*count)+extra>0?static_cast<long long>(*count)+extra:0LL);
    const long long needed=(tasks+tasksPerBlock-1)/tasksPerBlock;
    unsigned int selected=buckets.count;
    if(tasks){selected=0;while(selected+1<unsigned(buckets.count)&&buckets.blocks[selected]<needed)++selected;}
    return selected;
}
static __global__ void selectKernelGrid(const int* count,int tasksPerBlock,int extra,KernelGridBuckets buckets,cudaGraphConditionalHandle handle){
    if(threadIdx.x||blockIdx.x)return;
    cudaGraphSetConditional(handle,selectedKernelGrid(count,tasksPerBlock,extra,buckets));
}
struct KernelGridSelection {
    const int* count=nullptr;int tasksPerBlock=1,extra=0;
    KernelGridBuckets buckets{};cudaGraphConditionalHandle handle{};
    const int* valid=nullptr;const int* active=nullptr;const int* mode=nullptr;
};

struct KernelGridPublication {int count=0;KernelGridSelection selections[16]{};};
static __global__ void publishKernelGrids(KernelGridPublication publication){
    const int i=threadIdx.x;
    if(i<publication.count){
        const auto& s=publication.selections[i];
        const bool enabled=(!s.valid||*s.valid==1)&&(!s.active||*s.active!=0)&&(!s.mode||*s.mode!=2);
        cudaGraphSetConditional(s.handle,enabled?selectedKernelGrid(s.count,s.tasksPerBlock,s.extra,s.buckets):unsigned(s.buckets.count));
    }
}
}}
#endif
