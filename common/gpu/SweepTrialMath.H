#ifndef UGKP_GPU_SWEEP_TRIAL_MATH_H
#define UGKP_GPU_SWEEP_TRIAL_MATH_H
#include <cmath>
#include <limits>
#include <cfloat>
#ifdef __CUDACC__
#define UGKP_SWEEP_TRIAL_HD __host__ __device__
#else
#define UGKP_SWEEP_TRIAL_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
// Alpha is the existing line-search sequence 2^-backtrack, in [0,1].
// Halving before TwoSum prevents overflow without changing a range decision.
// Around DBL_MAX, the low part is compared with the rounding threshold of the
// native host long-double format. No floating-point value is stored in FP32.
UGKP_SWEEP_TRIAL_HD inline bool sweepTrialRangeFp64(double origin,double correction,double alpha){
    if(!std::isfinite(origin)||!std::isfinite(correction)||!std::isfinite(alpha))return false;
    const double a=origin*.5,b=(alpha*correction)*.5;
    const double sum=a+b,part=sum-a,error=(a-(sum-part))+(b-part);
    const double absolute=sum<0?-sum:sum,limit=std::numeric_limits<double>::max()*.5;
    if(absolute<limit)return true;if(absolute>limit||!std::isfinite(absolute))return false;
    const double low=sum<0?-error:error;
    const double threshold=::ldexp(1.,DBL_MAX_EXP-LDBL_MANT_DIG-2);
    return LDBL_MANT_DIG==DBL_MANT_DIG?low<threshold:low<=threshold;
}
struct SweepTrialCoordinate {double value=0;bool representable=false,changed=false;};
UGKP_SWEEP_TRIAL_HD inline SweepTrialCoordinate sweepTrialCoordinate(double origin,double correction,double alpha){
    SweepTrialCoordinate result;result.value=origin;
#ifdef __CUDA_ARCH__
    result.representable=sweepTrialRangeFp64(origin,correction,alpha);
#else
    const long double value=static_cast<long double>(origin)+static_cast<long double>(alpha)*correction;
    result.representable=std::isfinite(value)&&std::abs(value)<=std::numeric_limits<double>::max();
#endif
    if(result.representable){result.value=origin+alpha*correction;result.changed=result.value!=origin;}
    return result;
}
}}}
#undef UGKP_SWEEP_TRIAL_HD
#endif
