#include "mesh/Remap1D.H"
#include "gas/KernelSupport.cuh"
namespace chmt {
    namespace {
        __global__ void overlapKernel(const Real* oldEdges, int nOld, const GasQ* oldQ, const Real* newEdges,
            int nNew, GasQ* result, DeviceStatus* status) {
            int c = blockIdx.x*blockDim.x+threadIdx.x;
            if (c >= nNew || gasStopped(status))return;
            GasQ q;
            if (!remapCell1D(oldEdges, nOld, oldQ, newEdges, nNew, c, q)) {
                gasDeviceError(status, ErrorCode::InvalidInput, c, 0, ErrorLocation::Cell);
                return;
            }
            result[c] = q;
        }
    }
    int launchConservativeRemap1D(const double* oldEdges, int nOld, const GasQ* oldQ, const double* newEdges,
        int nNew, GasQ* result, DeviceStatus* status, void* stream) {
        if (!oldEdges || !newEdges || !oldQ || !result || !status || nOld <= 0 || nNew <= 0
            || oldQ == result)return static_cast<int>(cudaErrorInvalidValue);
        overlapKernel<<<(nNew+127)/128, 128, 0, static_cast<cudaStream_t>(stream)>>>(oldEdges, nOld, oldQ,
            newEdges, nNew, result, status);
        return launchError();
    }
}
