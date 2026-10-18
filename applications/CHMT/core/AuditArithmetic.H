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
    if(b>0&&a>bound-b)return std::numeric_limits<Real>::infinity();
    if(b<0&&a<-bound-b)return -std::numeric_limits<Real>::infinity();
    return a+b;
}
CHMT_HD inline Real auditDifference(Real a,Real b) { return auditAdd(a,-b); }
CHMT_HD inline Real auditProduct(Real a,Real b) {
    if(!finite(a)||!finite(b))return undefinedValue();
    const Real aa=absValue(a),bb=absValue(b);
    if(bb>1&&aa>std::numeric_limits<Real>::max()/bb)
        return ((a<0)!=(b<0))?-std::numeric_limits<Real>::infinity():std::numeric_limits<Real>::infinity();
    return a*b;
}
CHMT_HD inline Real auditTolerance(Real absolute,Real relative,Real scale) {
    return auditAdd(absolute,auditProduct(relative,scale));
}
}
#endif
