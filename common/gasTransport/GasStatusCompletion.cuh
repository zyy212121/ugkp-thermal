#pragma once
#include "GasCapabilities.H"
#include "../Diagnostics.H"
#include <cuda_runtime.h>

namespace ugkwp {
namespace gasStatusCompletion {
// Status consumption only: no thermodynamic/transport computation or mutation.
// Encoding cells before faces preserves the existing host scan's selection.
using Index=unsigned long long;
constexpr Index unavailableIndex=~Index(0);
static __global__ void findUnavailable(const int* cells,int cellCount,
    const int* faces,int faceCount,const ugkp::diagnostics::DeviceContext* diagnostics,
    Index* first) {
    if(ugkp::diagnostics::hasFatal(diagnostics))return;
    const Index count=Index(cellCount)+Index(faceCount);
    const Index stride=Index(blockDim.x)*gridDim.x;
    for(Index i=Index(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=stride) {
        const int code=i<Index(cellCount)?cells[i]:faces[i-Index(cellCount)];
        if(code==int(GasTransportCode::InvalidStorage))atomicMin(first,i);
    }
}
static __global__ void publishUnavailable(const Index* first,int cellCount,
    ugkp::diagnostics::DeviceContext* diagnostics,unsigned site,unsigned operation) {
    if(blockIdx.x||threadIdx.x||ugkp::diagnostics::hasFatal(diagnostics))return;
    const Index selected=*first;
    if(selected==unavailableIndex)return;
    const bool face=selected>=Index(cellCount);
    const auto location=static_cast<std::int64_t>(face?selected-Index(cellCount):selected);
    ugkp::diagnostics::recordFatal(diagnostics,ugkp::diagnostics::makeRecord(
        ugkp::diagnostics::Code::MissingOperator,site,operation,0,
        face?ugkp::diagnostics::EntityKind::Face:ugkp::diagnostics::EntityKind::Cell,
        location,-1,int(GasTransportCode::InvalidStorage)));
}
}

// The caller owns one scalar of scratch storage and the completion boundary.
// All three operations use its stream; no allocation, host readback or wait.
inline cudaError_t enqueueGasStatusCheck(const int* cells,int cellCount,
    const int* faces,int faceCount,ugkp::diagnostics::DeviceContext* diagnostics,
    unsigned long long* first,unsigned site,unsigned operation,cudaStream_t stream) {
    if(cellCount<0||faceCount<0||!diagnostics||!first
        ||(cellCount&&!cells)||(faceCount&&!faces))return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(first,0xff,sizeof(*first),stream);
    if(error!=cudaSuccess)return error;
    const auto count=static_cast<unsigned long long>(cellCount)+faceCount;
    if(count) {
        const auto needed=(count+255)/256;
        const int blocks=static_cast<int>(needed<65535?needed:65535);
        gasStatusCompletion::findUnavailable<<<blocks,256,0,stream>>>
            (cells,cellCount,faces,faceCount,diagnostics,first);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    gasStatusCompletion::publishUnavailable<<<1,1,0,stream>>>
        (first,cellCount,diagnostics,site,operation);
    return cudaGetLastError();
}
}
