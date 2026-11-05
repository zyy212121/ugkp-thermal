#ifndef CHMT_GPU_MATERIALREACTIONKERNELS_CUH
#define CHMT_GPU_MATERIALREACTIONKERNELS_CUH
#include "materials/MaterialReactionMath.H"
namespace chmt {
__global__ void advanceMaterialReactionsKernel(const SolidQ* base,const Real* volumes,
    int count,const PhysicsConfig* physics,Real interval,Real maxStep,Real cfl,
    int warningBudget,SolidQ* output,MaterialReactionStatus* status) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=count)return;
    status[cell]=advanceMaterialReactionsMath(base[cell],volumes[cell],*physics,
        interval,maxStep,cfl,warningBudget,output[cell]);
}
}
#endif
