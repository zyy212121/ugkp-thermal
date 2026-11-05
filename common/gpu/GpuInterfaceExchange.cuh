#ifndef UGKP_GPU_INTERFACE_EXCHANGE_CUH
#define UGKP_GPU_INTERFACE_EXCHANGE_CUH
namespace ugkp { namespace interfaceExchange {
// Each destination owns its row. Stable CSR source order retains the original
// face-loop summation, including cancellation, without floating-point atomics.
template<class Real>
__global__ void sumMappedFaceValues(int rows,const int* offsets,
    const int* sourceIndices,const Real* source,Real* destination)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    Real value=0;
    for(int entry=offsets[row];entry<offsets[row+1];++entry)
        value+=source[sourceIndices[entry]];
    destination[row]=value;
}
}}
#endif
