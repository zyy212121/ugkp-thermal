#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define UGKP_CLOCK_HD __host__ __device__
#else
#define UGKP_CLOCK_HD
#endif
namespace ugkp {
// A local clock is kept separately from its absolute output timestamp. Two
// FP64 components retain sub-timestamp-ULP steps on CPU and CUDA alike.
struct CompensatedTime {
    double sum=0,correction=0;
    UGKP_CLOCK_HD CompensatedTime():sum(0),correction(0){}
    UGKP_CLOCK_HD explicit CompensatedTime(double value):sum(value){}
    UGKP_CLOCK_HD static double addRounded(double a,double b){
#if defined(__CUDA_ARCH__)
        return __dadd_rn(a,b);
#else
        return a+b;
#endif
    }
    UGKP_CLOCK_HD static double subtractRounded(double a,double b){
#if defined(__CUDA_ARCH__)
        return __dsub_rn(a,b);
#else
        return a-b;
#endif
    }
    UGKP_CLOCK_HD void add(double value){
        const double next=addRounded(sum,value);
        const double residual=::fabs(sum)>=::fabs(value)
            ?addRounded(subtractRounded(sum,next),value)
            :addRounded(subtractRounded(value,next),sum);
        const double tail=addRounded(correction,residual);
        const double total=addRounded(next,tail);
        correction=subtractRounded(tail,subtractRounded(total,next));sum=total;
    }
    UGKP_CLOCK_HD double value()const{return addRounded(sum,correction);}
};
UGKP_CLOCK_HD inline CompensatedTime operator+(CompensatedTime a,double b){a.add(b);return a;}
UGKP_CLOCK_HD inline CompensatedTime operator-(CompensatedTime a,const CompensatedTime& b){a.add(-b.sum);a.add(-b.correction);return a;}
UGKP_CLOCK_HD inline bool operator<(const CompensatedTime& a,const CompensatedTime& b){return a.sum<b.sum||(a.sum==b.sum&&a.correction<b.correction);}
UGKP_CLOCK_HD inline bool operator==(const CompensatedTime& a,const CompensatedTime& b){return a.sum==b.sum&&a.correction==b.correction;}
UGKP_CLOCK_HD inline double compensatedTimeRatio(CompensatedTime numerator,CompensatedTime denominator){
    const double first=numerator.sum/denominator.sum;
    // One compensated quotient correction suffices for the public FP64 ratio.
    const double remainder=CompensatedTime::addRounded(::fma(-first,denominator.sum,numerator.sum),
        CompensatedTime::subtractRounded(numerator.correction,first*denominator.correction));
    return CompensatedTime::addRounded(first,remainder/denominator.sum);
}
UGKP_CLOCK_HD inline double absoluteTime(double origin,CompensatedTime local){
    CompensatedTime value(origin);value.add(local.sum);value.add(local.correction);return value.value();
}
} // namespace ugkp
#undef UGKP_CLOCK_HD
