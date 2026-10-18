#ifndef CHMT_CORE_AUDITARITHMETIC_H
#define CHMT_CORE_AUDITARITHMETIC_H
#include "Types.H"
namespace chmt {
// Diagnostic reductions must not raise OpenFOAM's enabled overflow/invalid traps.
// An infinity/NaN here describes an unrepresentable audit, never a repaired state.
CHMT_HD inline Real auditAdd(Real a,Real b) {
    const Real bound=std::numeric_limits<Real>::max();
    if(!finite(a)||!finite(b)) {
        if(a!=a||b!=b||(!finite(a)&&!finite(b)&&((a<0)!=(b<0))))return undefinedValue();
        return !finite(a)?a:b;
    }
    if(((a>0&&b>0)||(a<0&&b<0))
        &&(absValue(a)>bound*Real(.5)||absValue(b)>bound*Real(.5))) {
        // max-b may round upward at the last ULP. Binary scaling performs
        // the same rounded addition without ever forming an overflowing sum.
        const Real half=a*Real(.5)+b*Real(.5);
        if(absValue(half)>bound*Real(.5))return a<0
            ?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity();
        return half*Real(2);
    }
    return a+b;
}
CHMT_HD inline Real auditDifference(Real a,Real b) { return auditAdd(a,-b); }
CHMT_HD inline Real auditProduct(Real a,Real b) {
    if(!finite(a)||!finite(b))return undefinedValue();
    const Real aa=absValue(a),bb=absValue(b);
    if(aa<=1||bb<=1)return a*b;
    int ea=0,eb=0,em=0;
    const Real ma=::frexp(aa,&ea),mb=::frexp(bb,&eb);
    const Real mantissa=ma*mb; // [1/4,1), so this multiplication cannot overflow.
    (void)::frexp(mantissa,&em);
    const int exponent=ea+eb+em;
    if(exponent>std::numeric_limits<Real>::max_exponent)
        return ((a<0)!=(b<0))?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity();
    if(exponent==std::numeric_limits<Real>::max_exponent) {
        // max/b may round upward. Scale the already rounded mantissa here;
        // a result rounding to finite MAX must remain finite, not a warning.
        const Real result=::ldexp(mantissa,ea+eb);
        return ((a<0)!=(b<0))?-result:result;
    }
    return a*b;
}
// Quotients used only by diagnostics/tolerances, including explicit IEEE
// audit sentinels. Never execute zero division or an overflowing division.
CHMT_HD inline Real auditQuotient(Real numerator,Real denominator) {
    if(numerator!=numerator||denominator!=denominator)return undefinedValue();
    const bool negative=::copysign(Real(1),numerator)!=::copysign(Real(1),denominator);
    const Real infinity=std::numeric_limits<Real>::infinity();
    if(denominator==0)return numerator==0?undefinedValue():(negative?-infinity:infinity);
    if(!finite(numerator))return !finite(denominator)?undefinedValue():(negative?-infinity:infinity);
    if(!finite(denominator)||numerator==0)return ::copysign(Real(0),negative?Real(-1):Real(1));
    int en=0,ed=0,extra=0;
    const Real mn=::frexp(absValue(numerator),&en),md=::frexp(absValue(denominator),&ed);
    const Real mantissa=::frexp(mn/md,&extra);
    const int exponent=en-ed+extra;
    const int maximum=std::numeric_limits<Real>::max_exponent;
    const Real maximumMantissa=::ldexp(std::numeric_limits<Real>::max(),-maximum);
    if(exponent>maximum||(exponent==maximum&&mantissa>maximumMantissa))return negative?-infinity:infinity;
    return numerator/denominator;
}
// Keep an unrepresentable denominator product from destroying a representable
// audit quotient. The usual path keeps its original division order.
CHMT_HD inline Real auditQuotientProduct(Real numerator,Real first,Real second) {
    const Real product=auditProduct(first,second);
    if(finite(product)&&product!=0)return auditQuotient(numerator,product);
    if(!finite(numerator)||!finite(first)||!finite(second)||first==0||second==0)
        return auditQuotient(numerator,product);
    if(numerator==0)return Real(0);
    int en=0,ef=0,es=0,extra=0;
    const Real mn=::frexp(absValue(numerator),&en),mf=::frexp(absValue(first),&ef),ms=::frexp(absValue(second),&es);
    const Real mantissa=::frexp((mn/mf)/ms,&extra);
    const int exponent=en-ef-es+extra,maximum=std::numeric_limits<Real>::max_exponent;
    const bool negative=(numerator<0)!=((first<0)!=(second<0));
    if(exponent>maximum||(exponent==maximum&&mantissa>::ldexp(std::numeric_limits<Real>::max(),-maximum)))
        return negative?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity();
    const Real result=::ldexp(mantissa,exponent);return negative?-result:result;
}
CHMT_HD inline Real auditTolerance(Real absolute,Real relative,Real scale) {
    return auditAdd(absolute,auditProduct(relative,scale));
}
}
#endif
