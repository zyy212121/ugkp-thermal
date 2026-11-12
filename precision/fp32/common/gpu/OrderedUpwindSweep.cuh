#ifndef UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#define UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#include "RadiationPrecision.H"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
namespace ugkp { namespace gpu {
constexpr RadiationReal upwindIntensityRoundoff=256*std::numeric_limits<RadiationReal>::epsilon();
struct OrderedUpwindDevice {
    int cells=0;
    const int *offsets=nullptr,*neighbour=nullptr,*subtract=nullptr,*residualOrder=nullptr,*levelCells=nullptr;
    const RadiationReal* flux=nullptr;
};
struct UpwindIntensityFailure {int cell;RadiationReal value;};
__global__ void snapshotUpwind(int cells,const RadiationReal* current,RadiationReal* previous,const int* active=nullptr){
    if(active&&!*active)return;
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell<cells)previous[cell]=current[cell];
}
__device__ inline void orderedUpwindCell(int cell,OrderedUpwindDevice matrix,
    const RadiationReal* diagonal,const RadiationReal* source,const RadiationReal* previous,RadiationReal* current,
    RadiationReal reference,UpwindIntensityFailure* failures,unsigned int* firstFailure,
    const RadiationReal* deviceReference){
    RadiationReal rhs=source[cell];
    for(int i=matrix.offsets[cell];i<matrix.offsets[cell+1];++i){
        const int neighbour=matrix.neighbour[i];const RadiationReal value=neighbour<cell?current[neighbour]:previous[neighbour];
        if(matrix.subtract[i])rhs-=matrix.flux[i]*value;else rhs+=matrix.flux[i]*value;
    }
    RadiationReal value=rhs/diagonal[cell];
    if(deviceReference)reference=*deviceReference;
    // Same physical intensity consumption and roundoff bound as the native DO.
    if(!std::isfinite(value)){failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}
    if(value<0){const RadiationReal magnitude=fabs(reference);const RadiationReal scale=magnitude>1?magnitude:1;
        if(value>=-upwindIntensityRoundoff*scale)value=0;
        else {failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}}
    current[cell]=value;failures[cell]={-1,0};
}
__global__ void orderedUpwindLevel(int begin,int end,OrderedUpwindDevice matrix,
    const RadiationReal* diagonal,const RadiationReal* source,const RadiationReal* previous,RadiationReal* current,
    RadiationReal reference,UpwindIntensityFailure* failures,unsigned int* firstFailure=nullptr,
    const RadiationReal* deviceReference=nullptr,const int* active=nullptr){
    if(active&&!*active)return;
    const int row=begin+blockIdx.x*blockDim.x+threadIdx.x;if(row>=end)return;
    orderedUpwindCell(matrix.levelCells[row],matrix,diagonal,source,previous,current,reference,failures,firstFailure,deviceReference);
}
// A single CTA traverses a consecutive run of narrow dependency levels.
// Every thread reaches the barrier, including rows whose physical check failed.
// Kernel boundaries on either side retain the original wide-level dependencies.
__global__ void orderedUpwindNarrowLevels(int firstLevel,int endLevel,const int* levelOffsets,
    OrderedUpwindDevice matrix,const RadiationReal* diagonal,const RadiationReal* source,
    const RadiationReal* previous,RadiationReal* current,RadiationReal reference,
    UpwindIntensityFailure* failures,unsigned int* firstFailure,const RadiationReal* deviceReference,
    const int* active){
    if(active&&!*active)return;
    for(int level=firstLevel;level<endLevel;++level){
        const int row=levelOffsets[level]+threadIdx.x;
        if(row<levelOffsets[level+1])
            orderedUpwindCell(matrix.levelCells[row],matrix,diagonal,source,previous,current,
                reference,failures,firstFailure,deviceReference);
        __syncthreads();
    }
}
__global__ void upwindResidual(OrderedUpwindDevice matrix,const RadiationReal* diagonal,const RadiationReal* source,
    const RadiationReal* current,RadiationReal* residual,RadiationReal* magnitude){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=matrix.cells)return;
    RadiationReal value=diagonal[cell]*current[cell]-source[cell];
    const RadiationReal a=fabs(diagonal[cell]*current[cell]),b=fabs(source[cell]);magnitude[cell]=a>b?a:b;
    // The native residual accumulates internal faces in global face order.
    for(int j=matrix.offsets[cell];j<matrix.offsets[cell+1];++j){const int i=matrix.residualOrder[j];
        if(matrix.subtract[i])value+=matrix.flux[i]*current[matrix.neighbour[i]];
        else value-=matrix.flux[i]*current[matrix.neighbour[i]];}
    residual[cell]=value;
}
}}
#endif
