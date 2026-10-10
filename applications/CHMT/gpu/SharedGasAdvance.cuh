#ifndef CHMT_GPU_SHAREDGASADVANCE_CUH
#define CHMT_GPU_SHAREDGASADVANCE_CUH
// Include in a CUDA TU after the actual common gas operators and its
// GasHostPolicy/error helpers. CHMT supplies storage and coupling callbacks;
// the only gas transport/stage implementation is the common host owner.
#include "../../../common/GpuGasAdvance.cuh"
namespace chmt {
template<class CommonGasHost>
int advanceSharedGasTransport(CommonGasHost* host,GasHostPolicy::Time interval,
    GasHostPolicy::Time sampleTime) {
    return advanceGasFluxStage(host,interval,sampleTime);
}
}
#endif
