#ifndef CHMT_GPU_SHAREDGASADVANCE_CUH
#define CHMT_GPU_SHAREDGASADVANCE_CUH



#include "../../../common/GpuGasAdvance.cuh"
namespace chmt {
template<class CommonGasHost>
int advanceSharedGasTransport(CommonGasHost* host,GasHostPolicy::Time interval,
    GasHostPolicy::Time sampleTime) {
    return advanceGasFluxStage(host,interval,sampleTime);
}
}
#endif
