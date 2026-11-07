#ifndef UGKP_GPU_SWEEP_RESIDUAL_MATH_H
#define UGKP_GPU_SWEEP_RESIDUAL_MATH_H
#include "PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_SWEEP_RESIDUAL_HD __host__ __device__
#else
#define UGKP_SWEEP_RESIDUAL_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
UGKP_SWEEP_RESIDUAL_HD inline double sweepResidualAbsolute(double x){return x<0?-x:x;}
// Dimensionless reporting cannot invalidate a representable physical residual.
// CUDA's correctly rounded FP64 divide is saturated at the same finite bound
// as the native extended-range quotient; no single precision is introduced.
UGKP_SWEEP_RESIDUAL_HD inline double sweepResidualRatio(double residual,double tolerance){
    if(!std::isfinite(residual)||residual<0||!(tolerance>0))return std::numeric_limits<double>::quiet_NaN();
    const double limit=std::numeric_limits<double>::max();
#ifdef __CUDA_ARCH__
    const double ratio=residual/tolerance;
#else
    const long double ratio=static_cast<long double>(residual)/tolerance;
#endif
    return ratio>limit?limit:static_cast<double>(ratio);
}
UGKP_SWEEP_RESIDUAL_HD inline double sweepResidualTolerance(double target,double actual,double scale,
    double absolute,double relative,const double* ceiling,int row){
    const double a=sweepResidualAbsolute(target),b=sweepResidualAbsolute(actual);
    const double requested=absolute+relative*(a>b?a:b);
    const double floor=32*std::numeric_limits<double>::epsilon()*scale;
    const double tolerance=requested>floor?requested:floor;
    return ceiling?(tolerance<ceiling[row]?tolerance:ceiling[row]):tolerance;
}
}}}
#undef UGKP_SWEEP_RESIDUAL_HD
#endif
