#ifndef UGKP_GPU_SCALAR_WORK_LAUNCH_CUH
#define UGKP_GPU_SCALAR_WORK_LAUNCH_CUH
#include <cuda_runtime.h>
#include <algorithm>
#include "WorkPlan.H"
namespace ugkp { namespace gpu {
struct ScalarWorkLaunch {int blocks=0,threads=1,workersPerBlock=1;};
// The configured CTA width remains exact (default 128). Independent expensive
// workers may occupy only a prefix of each CTA so a small pool can use multiple
// SMs. This is a static resource policy, not a timed kernel race. Collective
// wall teams bypass it and retain their fixed algorithmic team widths.
inline ScalarWorkLaunch scalarWorkShape(int workers,int threads,int sms,int perSm){
    ScalarWorkLaunch q;q.threads=threads;
    if(workers<=0)return q;
    const long long resident=std::max(1LL,static_cast<long long>(sms)*perSm);
    q.workersPerBlock=int(std::min(static_cast<long long>(threads),
        std::max(1LL,(static_cast<long long>(workers)+resident-1)/resident)));
    q.blocks=int((static_cast<long long>(workers)+q.workersPerBlock-1)/q.workersPerBlock);
    return q;
}
template<class Kernel>
cudaError_t scalarWorkLaunch(Kernel kernel,int workers,ScalarWorkLaunch& launch){
    launch={};if(workers<=0)return cudaSuccess;
    const int threads=workPlan().materialBlockThreads;
    int device=0;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    // Resource data is invariant for this kernel/device/configured CTA width.
    // Cache per host thread; changing worker count only changes the cheap shape.
    struct Resources {int device=-1,threads=0,sms=0,perSm=0;Kernel kernel{};};
    static thread_local Resources cached;
    if(cached.device!=device||cached.threads!=threads||cached.kernel!=kernel){
        int sms=0,perSm=0;
        error=cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device);if(error!=cudaSuccess)return error;
        error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm,kernel,threads,0);if(error!=cudaSuccess)return error;
        cached={device,threads,sms,perSm,kernel};
    }
    launch=scalarWorkShape(workers,threads,cached.sms,cached.perSm);
    return cudaSuccess;
}
}}
#endif
