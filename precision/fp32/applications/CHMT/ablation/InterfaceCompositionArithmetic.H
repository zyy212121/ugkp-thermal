#ifndef CHMT_ABLATION_INTERFACECOMPOSITIONARITHMETIC_H
#define CHMT_ABLATION_INTERFACECOMPOSITIONARITHMETIC_H
namespace composition_detail {
    // These arithmetic guards preserve the original operation when representable.
    // Trial equations never execute a host overflow/divide-by-zero before returning
    // a typed mathematical failure; the caller owns final error publication.

    CHMT_HD inline bool productFinite(PhysicalReal a,PhysicalReal b,PhysicalReal& result) {
        if(!finite(a)||!finite(b))return false;
        const PhysicalReal aa=energyArithmetic::absolute(a),bb=energyArithmetic::absolute(b),safe=std::sqrt(std::numeric_limits<PhysicalReal>::max())*PhysicalReal(.5);
        if(aa<=1||bb<=1||(aa<=safe&&bb<=safe)){result=a*b;return true;}
        // Same binary scaling as auditProduct. Scaling an already rounded
        // mantissa preserves a result that rounds to finite MAX, and never
        // executes the overflowing unscaled operation.
        int ea=0,eb=0,em=0;
        const PhysicalReal mantissa=std::frexp(aa,&ea)*std::frexp(bb,&eb);
        (void)std::frexp(mantissa,&em);
        const int exponent=ea+eb+em;
        if(exponent>std::numeric_limits<PhysicalReal>::max_exponent){result=(a<0)!=(b<0)?-std::numeric_limits<PhysicalReal>::infinity():std::numeric_limits<PhysicalReal>::infinity();return false;}
        const PhysicalReal magnitude=exponent==std::numeric_limits<PhysicalReal>::max_exponent?std::ldexp(mantissa,ea+eb):aa*bb;
        result=(a<0)!=(b<0)?-magnitude:magnitude;return true;
    }

    CHMT_HD inline bool quotientFinite(PhysicalReal a,PhysicalReal b,PhysicalReal& result) {
        if(!finite(a)||!finite(b)||b==0)return false;
        const PhysicalReal aa=energyArithmetic::absolute(a),bb=energyArithmetic::absolute(b),safe=std::sqrt(std::numeric_limits<PhysicalReal>::max())*PhysicalReal(.5);
        if(bb>=1||aa==0||(aa<=safe&&bb>=1/safe)){result=a/b;return true;}
        int ea=0,eb=0,em=0;
        const PhysicalReal mantissa=std::frexp(aa,&ea)/std::frexp(bb,&eb);
        (void)std::frexp(mantissa,&em);
        const int exponent=ea-eb+em;
        if(exponent>std::numeric_limits<PhysicalReal>::max_exponent){result=(a<0)!=(b<0)?-std::numeric_limits<PhysicalReal>::infinity():std::numeric_limits<PhysicalReal>::infinity();return false;}
        const PhysicalReal magnitude=exponent==std::numeric_limits<PhysicalReal>::max_exponent?std::ldexp(mantissa,ea-eb):aa/bb;
        result=(a<0)!=(b<0)?-magnitude:magnitude;return true;
    }
    // The factorization avoids an overflowing intermediate in C*D/Rref.  frexp
    // mantissas stay bounded; the final scalbn is preceded by a range check.

    CHMT_HD inline bool diffusionScale(PhysicalReal pressure,PhysicalReal D,PhysicalReal T,PhysicalReal distance,
    PhysicalReal Rref,PhysicalReal& result) {
        if(D==0){
            result=0;
            return true;
        }
        PhysicalReal denominator=0,rho=0,weighted=0;
        if(productFinite(Rref,T,denominator)&&denominator>0
        &&quotientFinite(pressure,denominator,rho)&&rho>0
        &&productFinite(rho,D,weighted)&&weighted>0
        &&quotientFinite(weighted,distance,result)&&result>0)return true;
        int ep=0,ed=0,et=0,ex=0,er=0;
        PhysicalReal m=std::frexp(pressure,&ep)*std::frexp(D,&ed)
        /(std::frexp(T,&et)*std::frexp(distance,&ex)*std::frexp(Rref,&er));
        int em=0;
        m=std::frexp(m,&em);
        int exponent=ep+ed-et-ex-er+em;
        if(exponent>std::numeric_limits<PhysicalReal>::max_exponent || (exponent==std::numeric_limits<PhysicalReal>::max_exponent && m>std::ldexp(std::numeric_limits<PhysicalReal>::max(),-std::numeric_limits<PhysicalReal>::max_exponent)))return false;
        result=exponent < (std::numeric_limits<PhysicalReal>::min_exponent-std::numeric_limits<PhysicalReal>::digits) ? 0 : std::scalbn(m,exponent);
        return finite(result);
    }

    // R varies with each candidate. Share its EOS division across species while
    // preserving diffusionScale's operation order and guarded extreme-range path.
    struct DiffusionState {
        PhysicalReal pressure,T,distance,R,rho=0;
        bool attempted=false,ready=false;
        CHMT_HD DiffusionState(PhysicalReal pressure_,PhysicalReal T_,PhysicalReal distance_,PhysicalReal R_)
            :pressure(pressure_),T(T_),distance(distance_),R(R_){}
        CHMT_HD bool scale(PhysicalReal D,PhysicalReal& result) {
            if(D==0){result=0;return true;}
            if(!attempted){
                attempted=true;PhysicalReal denominator=0;
                ready=productFinite(R,T,denominator)&&denominator>0
                    &&quotientFinite(pressure,denominator,rho)&&rho>0;
            }
            PhysicalReal weighted=0;
            if(ready&&productFinite(rho,D,weighted)&&weighted>0
                &&quotientFinite(weighted,distance,result)&&result>0)return true;
            return diffusionScale(pressure,D,T,distance,R,result);
        }
    };

    CHMT_HD inline bool sumFinite(PhysicalReal a,PhysicalReal b,PhysicalReal& result) {
        const PhysicalReal maximum=std::numeric_limits<PhysicalReal>::max();
        if(!finite(a)||!finite(b))return false;
        if(((a>0&&b>0)||(a<0&&b<0)) && (energyArithmetic::absolute(a)>maximum*PhysicalReal(.5)||energyArithmetic::absolute(b)>maximum*PhysicalReal(.5))){
            PhysicalReal half=a*PhysicalReal(.5)+b*PhysicalReal(.5);
            if(energyArithmetic::absolute(half)>maximum*PhysicalReal(.5)){result=a<0?-std::numeric_limits<PhysicalReal>::infinity():std::numeric_limits<PhysicalReal>::infinity();return false;}
            result=half*2;return true;
        }
        result=a+b;return true;
    }

    CHMT_HD inline bool scaledSum(const PhysicalReal* values,const PhysicalReal* factors,PhysicalReal& result) {
        PhysicalReal direct=0;
        bool represented=true;
        for(int i=0;i<Ns;++i){
            PhysicalReal term=values[i];
            if(factors&&!productFinite(term,factors[i],term)){
                represented=false;
                break;
            }
            if(!sumFinite(direct,term,direct)){
                represented=false;
                break;
            }
        }
        if(represented){
            result=direct;
            return true;
        }
        PhysicalReal scale=0;
        for(int i=0;i<Ns;++i)scale=energyArithmetic::maximum(scale,energyArithmetic::absolute(values[i]));
        if(scale==0){
            result=0;
            return true;
        }
        PhysicalReal sum=0;
        for(int i=0;i<Ns;++i){
            PhysicalReal term=values[i]/scale;
            if(factors&&!productFinite(term,factors[i],term))return false;
            if(!sumFinite(sum,term,sum))return false;
        }
        return productFinite(scale,sum,result);
    }
}
namespace surfaceEnergyArithmetic {
// Diagnostic reductions must not raise OpenFOAM's enabled overflow/invalid traps.
// An infinity/NaN here describes an unrepresentable audit, never a repaired state.
CHMT_HD inline EnergyReal auditAdd(EnergyReal a,EnergyReal b) {
    const EnergyReal bound=std::numeric_limits<EnergyReal>::max();
    if(!finite(a)||!finite(b)) {
        if(a!=a||b!=b||(!finite(a)&&!finite(b)&&((a<0)!=(b<0))))return std::numeric_limits<EnergyReal>::quiet_NaN();
        return !finite(a)?a:b;
    }
    if(((a>0&&b>0)||(a<0&&b<0))
        &&(energyArithmetic::absolute(a)>bound*EnergyReal(.5)||energyArithmetic::absolute(b)>bound*EnergyReal(.5))) {
        // max-b may round upward at the last ULP. Binary scaling performs
        // the same rounded addition without ever forming an overflowing sum.
        const EnergyReal half=a*EnergyReal(.5)+b*EnergyReal(.5);
        if(energyArithmetic::absolute(half)>bound*EnergyReal(.5))return a<0
            ?-std::numeric_limits<EnergyReal>::infinity():std::numeric_limits<EnergyReal>::infinity();
        return half*EnergyReal(2);
    }
    return a+b;
}
CHMT_HD inline EnergyReal auditDifference(EnergyReal a,EnergyReal b) { return auditAdd(a,-b); }
CHMT_HD inline EnergyReal auditProduct(EnergyReal a,EnergyReal b) {
    if(!finite(a)||!finite(b))return std::numeric_limits<EnergyReal>::quiet_NaN();
    const EnergyReal aa=energyArithmetic::absolute(a),bb=energyArithmetic::absolute(b);
    if(aa<=1||bb<=1)return a*b;
    int ea=0,eb=0,em=0;
    const EnergyReal ma=::frexpf(aa,&ea),mb=::frexpf(bb,&eb);
    const EnergyReal mantissa=ma*mb; // [1/4,1), so this multiplication cannot overflow.
    (void)::frexpf(mantissa,&em);
    const int exponent=ea+eb+em;
    if(exponent>std::numeric_limits<EnergyReal>::max_exponent)
        return ((a<0)!=(b<0))?-std::numeric_limits<EnergyReal>::infinity():std::numeric_limits<EnergyReal>::infinity();
    if(exponent==std::numeric_limits<EnergyReal>::max_exponent) {
        // max/b may round upward. Scale the already rounded mantissa here;
        // a result rounding to finite MAX must remain finite, not a warning.
        const EnergyReal result=::ldexpf(mantissa,ea+eb);
        return ((a<0)!=(b<0))?-result:result;
    }
    return a*b;
}
// Quotients used only by diagnostics/tolerances, including explicit IEEE
// audit sentinels. Never execute zero division or an overflowing division.
CHMT_HD inline EnergyReal auditQuotient(EnergyReal numerator,EnergyReal denominator) {
    if(numerator!=numerator||denominator!=denominator)return std::numeric_limits<EnergyReal>::quiet_NaN();
    const bool negative=::copysignf(EnergyReal(1),numerator)!=::copysignf(EnergyReal(1),denominator);
    const EnergyReal infinity=std::numeric_limits<EnergyReal>::infinity();
    if(denominator==0)return numerator==0?std::numeric_limits<EnergyReal>::quiet_NaN():(negative?-infinity:infinity);
    if(!finite(numerator))return !finite(denominator)?std::numeric_limits<EnergyReal>::quiet_NaN():(negative?-infinity:infinity);
    if(!finite(denominator)||numerator==0)return ::copysignf(EnergyReal(0),negative?EnergyReal(-1):EnergyReal(1));
    int en=0,ed=0,extra=0;
    const EnergyReal mn=::frexpf(energyArithmetic::absolute(numerator),&en),md=::frexpf(energyArithmetic::absolute(denominator),&ed);
    const EnergyReal mantissa=::frexpf(mn/md,&extra);
    const int exponent=en-ed+extra;
    const int maximum=std::numeric_limits<EnergyReal>::max_exponent;
    const EnergyReal maximumMantissa=::ldexpf(std::numeric_limits<EnergyReal>::max(),-maximum);
    if(exponent>maximum||(exponent==maximum&&mantissa>maximumMantissa))return negative?-infinity:infinity;
    return numerator/denominator;
}
// Keep an unrepresentable denominator product from destroying a representable
// audit quotient. The usual path keeps its original division order.
CHMT_HD inline EnergyReal auditQuotientProduct(EnergyReal numerator,EnergyReal first,EnergyReal second) {
    const EnergyReal product=auditProduct(first,second);
    if(finite(product)&&product!=0)return auditQuotient(numerator,product);
    if(!finite(numerator)||!finite(first)||!finite(second)||first==0||second==0)
        return auditQuotient(numerator,product);
    if(numerator==0)return EnergyReal(0);
    int en=0,ef=0,es=0,extra=0;
    const EnergyReal mn=::frexpf(energyArithmetic::absolute(numerator),&en),mf=::frexpf(energyArithmetic::absolute(first),&ef),ms=::frexpf(energyArithmetic::absolute(second),&es);
    const EnergyReal mantissa=::frexpf((mn/mf)/ms,&extra);
    const int exponent=en-ef-es+extra,maximum=std::numeric_limits<EnergyReal>::max_exponent;
    const bool negative=(numerator<0)!=((first<0)!=(second<0));
    if(exponent>maximum||(exponent==maximum&&mantissa>::ldexpf(std::numeric_limits<EnergyReal>::max(),-maximum)))
        return negative?-std::numeric_limits<EnergyReal>::infinity():std::numeric_limits<EnergyReal>::infinity();
    const EnergyReal result=::ldexpf(mantissa,exponent);return negative?-result:result;
}
CHMT_HD inline EnergyReal auditTolerance(EnergyReal absolute,EnergyReal relative,EnergyReal scale) {
    return auditAdd(absolute,auditProduct(relative,scale));
}
}
#endif
