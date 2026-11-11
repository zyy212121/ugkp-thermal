#ifndef UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#define UGKP_GPU_ORDERED_UPWIND_SWEEP_CUH
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#ifndef UGKP_ORDERED_RECOVERY_NOTE
#define UGKP_ORDERED_RECOVERY_NOTE() ((void)0)
#endif
namespace ugkp { namespace gpu {
constexpr double upwindIntensityRoundoff=256*std::numeric_limits<double>::epsilon();
template<class Real> struct OrderedUpwindDeviceT {
    int cells=0;
    const int *offsets=nullptr,*neighbour=nullptr,*subtract=nullptr,*residualOrder=nullptr,*levelCells=nullptr;
    const Real* flux=nullptr;
    const double* trueFlux=nullptr;
    const int* fluxAddress=nullptr;
};
using OrderedUpwindDevice=OrderedUpwindDeviceT<double>;
template<class Real> struct UpwindCorrectionState {
    Real *values=nullptr,*low=nullptr;
    const double *offset=nullptr,*diagonal=nullptr,*residualMagnitude=nullptr;
    Real* correctionSource=nullptr;
};
struct UpwindIntensityFailure {int cell;double value;};
template<class Real> __global__ void snapshotUpwind(int cells,const Real* current,Real* previous,const int* active=nullptr){
    if(active&&!*active)return;
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell<cells)previous[cell]=current[cell];
}
template<class Real> __device__ inline void orderedUpwindCell(int cell,OrderedUpwindDeviceT<Real> matrix,
    const Real* diagonal,const Real* source,const Real* previous,Real* current,
    double reference,UpwindIntensityFailure* failures,unsigned int* firstFailure,
    const double* deviceReference,UpwindCorrectionState<Real> correction={}){
    Real rhs=source[cell];
    for(int i=matrix.offsets[cell];i<matrix.offsets[cell+1];++i){
        const int neighbour=matrix.neighbour[i];const Real value=neighbour<cell?current[neighbour]:(correction.values?Real(0):previous[neighbour]);
        if(matrix.subtract[i])rhs-=matrix.flux[i]*value;else rhs+=matrix.flux[i]*value;
    }
    const Real increment=rhs/diagonal[cell];
    const double base=correction.values?double(correction.values[cell])+double(correction.low[cell]):0;
    const double origin=correction.offset?*correction.offset:0;
    double value=correction.values?origin+(base+double(increment)):double(increment);
    // A rounded correction can overshoot zero even when the native positive
    // intensity row is valid. Recover that row in original precision before
    // the existing physical-consumption checks, rather than rejecting it.
    if(correction.values&&(value<0||!std::isfinite(value))){
        UGKP_ORDERED_RECOVERY_NOTE();
        double exact=copysign(correction.residualMagnitude[cell],double(correction.correctionSource[cell]));
        for(int i=matrix.offsets[cell];i<matrix.offsets[cell+1];++i){
            const int n=matrix.neighbour[i];
            if(n<cell){
                const double flux=matrix.trueFlux[matrix.fluxAddress?matrix.fluxAddress[i]:i];
                const double upstream=double(current[n])+double(correction.correctionSource[n]);
                if(matrix.subtract[i])exact-=flux*upstream;else exact+=flux*upstream;
            }
        }
        value=origin+(base+exact/correction.diagonal[cell]);
    }
    if(deviceReference)reference=*deviceReference;
    // Same physical intensity consumption and roundoff bound as the native DO.
    if(!std::isfinite(value)){failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}
    if(value<0){const double magnitude=fabs(reference);const double scale=magnitude>1?magnitude:1;
        if(value>=-upwindIntensityRoundoff*scale)value=0;
        else {failures[cell]={cell,value};if(firstFailure)atomicMin(firstFailure,unsigned(cell));return;}}
    if(correction.values){
        const double delta=value-origin;const Real high=Real(delta);
        current[cell]=Real(delta-base);
        // This RHS entry has been consumed. Reuse it for the correction's low
        // component, so a later recovery row reads original-order increments.
        correction.correctionSource[cell]=Real((delta-base)-double(current[cell]));
        correction.values[cell]=high;correction.low[cell]=Real(delta-double(high));
    }else current[cell]=Real(value);
    failures[cell]={-1,0};
}
template<class Real> __global__ void orderedUpwindLevel(int begin,int end,OrderedUpwindDeviceT<Real> matrix,
    const Real* diagonal,const Real* source,const Real* previous,Real* current,
    double reference,UpwindIntensityFailure* failures,unsigned int* firstFailure=nullptr,
    const double* deviceReference=nullptr,const int* active=nullptr,UpwindCorrectionState<Real> correction={}){
    if(active&&!*active)return;
    const int row=begin+blockIdx.x*blockDim.x+threadIdx.x;if(row>=end)return;
    orderedUpwindCell(matrix.levelCells[row],matrix,diagonal,source,previous,current,reference,failures,firstFailure,deviceReference,correction);
}
template<class Real> __global__ void upwindResidual(OrderedUpwindDeviceT<Real> matrix,const Real* diagonal,const Real* source,
    const Real* current,double* residual,double* magnitude,const Real* low=nullptr,
    const double* trueDiagonal=nullptr,const double* trueSource=nullptr,const double* offset=nullptr){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=matrix.cells)return;
    const double d=trueDiagonal?trueDiagonal[cell]:double(diagonal[cell]);
    const double s=trueSource?trueSource[cell]:double(source[cell]);
    const double origin=offset?*offset:0;
    const double x=origin+(double(current[cell])+(low?double(low[cell]):0));
    double value=d*x-s;
    for(int j=matrix.offsets[cell];j<matrix.offsets[cell+1];++j){const int i=matrix.residualOrder[j],n=matrix.neighbour[i];
        const double flux=matrix.trueFlux?matrix.trueFlux[matrix.fluxAddress?matrix.fluxAddress[i]:i]:double(matrix.flux[i]);
        const double upstream=origin+(double(current[n])+(low?double(low[n]):0));
        if(matrix.subtract[i])value+=flux*upstream;else value-=flux*upstream;}
    const double a=fabs(d*x),b=fabs(s);magnitude[cell]=a>b?a:b;
    residual[cell]=value;
}
// Signed FP32 correction sweeps retain the native ordered-GS dependency.
// The low component exists only in the current direction workspace. Physical
// intensity validation remains at orderedUpwindCell's original check point.
template<class Real> __global__ void prepareUpwindCorrection(int cells,Real* correction,const int* active){
    if(!*active)return;const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells)return;
    correction[cell]=0;
}
}}
#endif
