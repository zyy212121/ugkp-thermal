#ifndef UGKP_SCALED_THERMAL_SUM_H
#define UGKP_SCALED_THERMAL_SUM_H
#include "../GpuPrecisionTypes.H"
#include <cmath>
#ifdef __CUDACC__
#define UGKP_SUM_HD __host__ __device__
#else
#define UGKP_SUM_HD
#endif
namespace ugkp { namespace gpu {
// Compatibility name for a single FP32 thermal sum. No compensation, exponent
// extension, carry, or secondary energy representation is retained.
struct ScaledThermalSum {
    GpuReal value=0;bool populated=false;
    UGKP_SUM_HD void add(GpuReal x){value+=x;populated=value!=0;}
    UGKP_SUM_HD void product(GpuReal a,GpuReal b){add(a*b);}
    UGKP_SUM_HD void product(GpuReal a,GpuReal b,GpuReal c){add((a*b)*c);}
    UGKP_SUM_HD void merge(const ScaledThermalSum& other){add(other.value);}
    UGKP_SUM_HD bool finite()const{return std::isfinite(value);}
    UGKP_SUM_HD ScaledThermalSum negated()const{auto out=*this;out.value=-out.value;return out;}
    UGKP_SUM_HD ScaledThermalSum magnitude()const{return value<0?negated():*this;}
    UGKP_SUM_HD ScaledThermalSum multiplied(GpuReal factor)const{ScaledThermalSum out;out.add(value*factor);return out;}
    UGKP_SUM_HD bool lessEqual(const ScaledThermalSum& other)const{return finite()&&other.finite()&&value<=other.value;}
    UGKP_SUM_HD GpuReal ratioTo(const ScaledThermalSum& other)const{return value/other.value;}
    // Widen only at an existing report/publication boundary.
    UGKP_SUM_HD double binary64()const{return static_cast<double>(value);}
    long double extended()const{return static_cast<long double>(value);}
};
}}
#undef UGKP_SUM_HD
#endif
