#include "Diagnostics.H"
__global__ void diagnosticLoad(const ugkp::diagnostics::DeviceContext* context, int* output) {
    *output=ugkp::diagnostics::hasFatal(context);
}
