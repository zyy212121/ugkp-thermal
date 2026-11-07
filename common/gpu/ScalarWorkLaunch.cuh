#ifndef UGKP_GPU_SCALAR_WORK_LAUNCH_CUH
#define UGKP_GPU_SCALAR_WORK_LAUNCH_CUH
#include <cuda_runtime.h>
#include <algorithm>
namespace ugkp { namespace gpu {
struct ScalarWorkLaunch {int blocks=0,threads=1;};
// For independent scalar workers with expensive, divergent recurrences. Spread
// small pools over the device before packing a full warp into each block. Large
// pools retain one-warp blocks. Worker indices and their private scratch are
// unchanged; this policy must not be used for block-collective kernels.
// Device occupancy supplies the available concurrency, never a case/model knob.
template<class Kernel>
cudaError_t scalarWorkLaunch(Kernel kernel,int workers,ScalarWorkLaunch& launch){
    launch={};if(workers<=0)return cudaSuccess;
    int device=0,sms=0,warp=0,perSm=0;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    error=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(error!=cudaSuccess)return error;
    error=cudaDeviceGetAttribute(&warp,cudaDevAttrWarpSize,device);if(error!=cudaSuccess)return error;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm,kernel,warp,0);if(error!=cudaSuccess)return error;
    const long long resident=std::max(1LL,static_cast<long long>(sms)*perSm);
    launch.threads=int(std::min(static_cast<long long>(warp),
        std::max(1LL,(static_cast<long long>(workers)+resident-1)/resident)));
    launch.blocks=int((static_cast<long long>(workers)+launch.threads-1)/launch.threads);
    return cudaSuccess;
}
}}
#endif
