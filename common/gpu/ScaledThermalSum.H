#ifndef UGKP_SCALED_THERMAL_SUM_H
#define UGKP_SCALED_THERMAL_SUM_H
#include <cmath>
#ifdef __CUDACC__
#define UGKP_SUM_HD __host__ __device__
#else
#define UGKP_SUM_HD
#endif
namespace ugkp { namespace gpu {
// FP64 compensated mantissas with an independent exponent. This preserves
// finite sums/products beyond binary64 range without relying on GPU long double.
struct ScaledThermalSum {
    double value=0,correction=0;int exponent=0;bool populated=false;
    UGKP_SUM_HD void addScaled(double term,int power){
        if(term==0)return;
        if(!populated){exponent=power;populated=true;}
        if(power>exponent){value=ldexp(value,exponent-power);correction=ldexp(correction,exponent-power);exponent=power;}
        const double x=ldexp(term,power-exponent),next=value+x;
        correction+=fabs(value)>=fabs(x)?(value-next)+x:(x-next)+value;value=next;
        // Re-centre after cancellation: otherwise a zero sum retains the old
        // huge exponent and a subsequent finite small term can underflow.
        const double combined=value+correction;
        const double tail=fabs(value)>=fabs(correction)?(value-combined)+correction:(correction-combined)+value;
        value=combined;correction=tail;
        if(value==0&&correction==0){populated=false;exponent=0;return;}
        int shift=0;frexp(fabs(value)>=fabs(correction)?value:correction,&shift);
        value=ldexp(value,-shift);correction=ldexp(correction,-shift);exponent+=shift;
    }
    UGKP_SUM_HD void add(double x){int e=0;const double m=frexp(x,&e);addScaled(m,e);}
    UGKP_SUM_HD void product(double a,double b){int ea=0,eb=0;const double ma=frexp(a,&ea),mb=frexp(b,&eb);addScaled(ma*mb,ea+eb);}
    UGKP_SUM_HD void product(double a,double b,double c){int ea=0,eb=0,ec=0;const double ma=frexp(a,&ea),mb=frexp(b,&eb),mc=frexp(c,&ec);addScaled((ma*mb)*mc,ea+eb+ec);}
    UGKP_SUM_HD void merge(const ScaledThermalSum& other){if(other.populated){addScaled(other.value,other.exponent);addScaled(other.correction,other.exponent);}}
    UGKP_SUM_HD bool finite()const{return std::isfinite(value)&&std::isfinite(correction);}
    UGKP_SUM_HD ScaledThermalSum negated()const{auto result=*this;result.value=-result.value;result.correction=-result.correction;return result;}
    UGKP_SUM_HD ScaledThermalSum magnitude()const{return value<0||(value==0&&correction<0)?negated():*this;}
    UGKP_SUM_HD ScaledThermalSum multiplied(double factor)const{
        ScaledThermalSum result;if(!populated)return result;
        int power=0;const double mantissa=frexp(factor,&power);
        result.addScaled(value*mantissa,exponent+power);result.addScaled(correction*mantissa,exponent+power);return result;
    }
    UGKP_SUM_HD bool lessEqual(const ScaledThermalSum& other)const{
        if(!finite()||!other.finite())return false;
        auto difference=*this;difference.merge(other.negated());
        return difference.value<0||(difference.value==0&&difference.correction<=0);
    }
    UGKP_SUM_HD double ratioTo(const ScaledThermalSum& denominator)const{
        return ldexp((value+correction)/(denominator.value+denominator.correction),exponent-denominator.exponent);
    }
    UGKP_SUM_HD double binary64()const{return ldexp(value+correction,exponent);}
    long double extended()const{return std::ldexp(static_cast<long double>(value)+static_cast<long double>(correction),exponent);}
};
}}
#undef UGKP_SUM_HD
#endif
