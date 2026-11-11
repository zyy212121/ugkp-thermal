#ifndef UGKP_GPU_FLOAT_JACOBI_PRECONDITIONER_H
#define UGKP_GPU_FLOAT_JACOBI_PRECONDITIONER_H
#include <cfloat>
#include <cmath>
#ifdef __CUDACC__
#define UGKP_JACOBI_HD __host__ __device__
#else
#define UGKP_JACOBI_HD
#endif
namespace ugkp { namespace gpu { namespace floatJacobi {
// FP32 diagonal copy, division and z workspace; the physical matrix and outer
// Krylov state remain FP64. No scaled exponent arrays or altered tolerances.
UGKP_JACOBI_HD inline float encode(double residual,float diagonal){
    const float r=static_cast<float>(residual);
    const float a=std::fabs(diagonal),b=std::fabs(r);
    // Exceptional rows continue through the original FP64 division. A NaN
    // is only a private workspace tag and is decoded before any arithmetic.
    if(!(a>=FLT_MIN&&a<=FLT_MAX)||(!(b>=FLT_MIN&&b<=FLT_MAX)&&residual!=0))return NAN;
    const float z=r/diagonal;
    return std::isfinite(z)&&(std::fabs(z)>=FLT_MIN||residual==0)?z:NAN;
}
UGKP_JACOBI_HD inline double decode(float encoded,double residual,double diagonal){
    return std::isfinite(encoded)?static_cast<double>(encoded):residual/diagonal;
}
}}}
#undef UGKP_JACOBI_HD
#endif
