#ifndef CHMT_CORE_HOST_AUDIT_REDUCTION_H
#define CHMT_CORE_HOST_AUDIT_REDUCTION_H
#include "core/Types.H"
#include "PstreamReduceOps.H"
#include <cmath>
#include <limits>
namespace chmt {


inline long double hostAuditAdd(long double a,long double b){
    const long double infinity=std::numeric_limits<long double>::infinity();
    if(std::isnan(a)||std::isnan(b))return std::numeric_limits<long double>::quiet_NaN();
    if(std::isinf(a))return std::isinf(b)&&std::signbit(a)!=std::signbit(b)?std::numeric_limits<long double>::quiet_NaN():a;
    if(std::isinf(b))return b;
    const long double limit=std::numeric_limits<long double>::max();
    if(b>0&&a>limit-b)return infinity;
    if(b<0&&a< -limit-b)return -infinity;
    return a+b;
}
inline Real hostAuditNarrow(long double value){
    if(std::isnan(value))return std::numeric_limits<Real>::quiet_NaN();
    const long double limit=std::numeric_limits<Real>::max();
    if(value>limit)return std::numeric_limits<Real>::infinity();
    if(value< -limit)return -std::numeric_limits<Real>::infinity();
    if(value!=0&&std::abs(value)<std::numeric_limits<Real>::denorm_min())
        return std::signbit(value)?-Real(0):Real(0);
    return static_cast<Real>(value);
}
namespace hostAuditDetail {
struct FlagsOr{unsigned operator()(unsigned a,unsigned b)const{return a|b;}};
inline long double scale(long double fraction,int exponent){
    if(fraction==0)return 0;
    int extra=0;fraction=std::frexp(fraction,&extra);
    const long long power=static_cast<long long>(exponent)+extra;
    const int largest=std::numeric_limits<long double>::max_exponent;
    const long double largestFraction=std::ldexp(std::numeric_limits<long double>::max(),-largest);
    if(power>largest||(power==largest&&std::abs(fraction)>largestFraction))
        return std::signbit(fraction)?-std::numeric_limits<long double>::infinity():std::numeric_limits<long double>::infinity();
    if(power<std::numeric_limits<long double>::min_exponent-std::numeric_limits<long double>::digits)return 0;
    return std::ldexp(fraction,static_cast<int>(power));
}
}



inline long double hostAuditGlobalSum(long double local,Foam::label comm=Foam::UPstream::worldComm){
    if(Foam::UPstream::nProcs(comm)==1)return local;
    unsigned flags=std::isnan(local)?1u:(std::isinf(local)?(std::signbit(local)?4u:2u):0u);
    Foam::reduce(flags,hostAuditDetail::FlagsOr(),Foam::Pstream::msgType(),comm);
    if((flags&1u)||((flags&2u)&&(flags&4u)))return std::numeric_limits<long double>::quiet_NaN();
    if(flags&2u)return std::numeric_limits<long double>::infinity();
    if(flags&4u)return -std::numeric_limits<long double>::infinity();
    int exponent=std::numeric_limits<int>::min();
    if(local!=0)std::frexp(local,&exponent);
    Foam::reduce(exponent,Foam::maxOp<int>(),Foam::Pstream::msgType(),comm);
    if(exponent==std::numeric_limits<int>::min())return 0;
    Real scaled=local==0?Real(0):hostAuditNarrow(std::ldexp(local,-exponent));

    Foam::reduce(scaled,Foam::sumOp<Real>(),Foam::Pstream::msgType(),comm);
    return hostAuditDetail::scale(static_cast<long double>(scaled),exponent);
}
}
#endif
