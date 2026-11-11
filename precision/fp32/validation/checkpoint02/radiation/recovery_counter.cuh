#ifndef UGKP_CHECKPOINT02_RECOVERY_COUNTER_CUH
#define UGKP_CHECKPOINT02_RECOVERY_COUNTER_CUH
#include <cuda_runtime.h>
__device__ unsigned long long checkpoint02RecoveredRows=0;
#define UGKP_ORDERED_RECOVERY_NOTE() atomicAdd(&::checkpoint02RecoveredRows,1ULL)
extern "C" unsigned long long checkpoint02RecoveryCount(){
    unsigned long long result=0;
    cudaMemcpyFromSymbol(&result,checkpoint02RecoveredRows,sizeof(result));
    return result;
}
#endif
