#ifndef UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#define UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
namespace ugkp { namespace gpu {
constexpr double upwindIntensityRoundoff=256*std::numeric_limits<double>::epsilon();
struct OrderedUpwindDevice {
    int cells=0;
    const int *offsets=nullptr,*neighbour=nullptr,*subtract=nullptr,*residualOrder=nullptr,*levelCells=nullptr;
    const double* flux=nullptr;
};
struct UpwindIntensityFailure {int cell;double value;};
__global__ void snapshotUpwind(int cells,const double* current,double* previous,const int* active=nullptr){
    if(active&&!*active)return;
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell<cells)previous[cell]=current[cell];
}
__device__ inline void orderedUpwindCell(int cell,OrderedUpwindDevice matrix,
    const double* diagonal,const double* source,const double* previous,double* current,
    double reference,UpwindIntensityFailure* failures,unsigned int* firstFailure,
    const double* deviceReference){
    double rhs=source[cell];
    for(int i=matrix.offsets[cell];i<matrix.offsets[cell+1];++i){
        const int neighbour=matrix.neighbour[i];const double value=neighbour<cell?current[neighbour]:previous[neighbour];
        if(matrix.subtract[i])rhs-=matrix.flux[i]*value;else rhs+=matrix.flux[i]*value;
    }
    double value=rhs/diagonal[cell];
    if(deviceReference)reference=*deviceReference;
    // Same physical intensity consumption and roundoff bound as the native DO.
    if(!std::isfinite(value)){failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}
    if(value<0){const double magnitude=fabs(reference);const double scale=magnitude>1?magnitude:1;
        if(value>=-upwindIntensityRoundoff*scale)value=0;
        else {failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}}
    current[cell]=value;failures[cell]={-1,0};
}
__global__ void orderedUpwindLevel(int begin,int end,OrderedUpwindDevice matrix,
    const double* diagonal,const double* source,const double* previous,double* current,
    double reference,UpwindIntensityFailure* failures,unsigned int* firstFailure=nullptr,
    const double* deviceReference=nullptr,const int* active=nullptr){
    if(active&&!*active)return;
    const int row=begin+blockIdx.x*blockDim.x+threadIdx.x;if(row>=end)return;
    orderedUpwindCell(matrix.levelCells[row],matrix,diagonal,source,previous,current,reference,failures,firstFailure,deviceReference);
}
__global__ void upwindResidual(OrderedUpwindDevice matrix,const double* diagonal,const double* source,
    const double* current,double* residual,double* magnitude){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=matrix.cells)return;
    double value=diagonal[cell]*current[cell]-source[cell];
    const double a=fabs(diagonal[cell]*current[cell]),b=fabs(source[cell]);magnitude[cell]=a>b?a:b;
    // The native residual accumulates internal faces in global face order.
    for(int j=matrix.offsets[cell];j<matrix.offsets[cell+1];++j){const int i=matrix.residualOrder[j];
        if(matrix.subtract[i])value+=matrix.flux[i]*current[matrix.neighbour[i]];
        else value-=matrix.flux[i]*current[matrix.neighbour[i]];}
    residual[cell]=value;
}
}}
#endif
