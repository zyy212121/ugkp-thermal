#ifndef CHMT_ABLATION_INTERFACECOMPOSITIONARITHMETIC_H
#define CHMT_ABLATION_INTERFACECOMPOSITIONARITHMETIC_H
namespace composition_detail {
    // These arithmetic guards preserve the original operation when representable.
    // Trial equations never execute a host overflow/divide-by-zero before returning
    // a typed mathematical failure; the caller owns final error publication.

    CHMT_HD inline bool productFinite(Real a,Real b,Real& result) {
        if(!finite(a)||!finite(b))return false;
        const Real aa=absValue(a),bb=absValue(b),safe=6.7039039649712985e153;
        if(aa<=1||bb<=1||(aa<=safe&&bb<=safe)){result=a*b;return true;}
        // Same binary scaling as auditProduct. Scaling an already rounded
        // mantissa preserves a result that rounds to finite MAX, and never
        // executes the overflowing unscaled operation.
        int ea=0,eb=0,em=0;
        const Real mantissa=::frexp(aa,&ea)*::frexp(bb,&eb);
        (void)::frexp(mantissa,&em);
        const int exponent=ea+eb+em;
        if(exponent>1024){result=(a<0)!=(b<0)?-HUGE_VAL:HUGE_VAL;return false;}
        const Real magnitude=exponent==1024?::ldexp(mantissa,ea+eb):aa*bb;
        result=(a<0)!=(b<0)?-magnitude:magnitude;return true;
    }

    CHMT_HD inline bool quotientFinite(Real a,Real b,Real& result) {
        if(!finite(a)||!finite(b)||b==0)return false;
        const Real aa=absValue(a),bb=absValue(b),safe=6.7039039649712985e153;
        if(bb>=1||aa==0||(aa<=safe&&bb>=1/safe)){result=a/b;return true;}
        int ea=0,eb=0,em=0;
        const Real mantissa=::frexp(aa,&ea)/::frexp(bb,&eb);
        (void)::frexp(mantissa,&em);
        const int exponent=ea-eb+em;
        if(exponent>1024){result=(a<0)!=(b<0)?-HUGE_VAL:HUGE_VAL;return false;}
        const Real magnitude=exponent==1024?::ldexp(mantissa,ea-eb):aa/bb;
        result=(a<0)!=(b<0)?-magnitude:magnitude;return true;
    }
    // The factorization avoids an overflowing intermediate in C*D/Rref.  frexp
    // mantissas stay bounded; the final scalbn is preceded by a range check.

    CHMT_HD inline bool diffusionScale(Real pressure,Real D,Real T,Real distance,
    Real Rref,Real& result) {
        if(D==0){
            result=0;
            return true;
        }
        Real denominator=0,rho=0,weighted=0;
        if(productFinite(Rref,T,denominator)&&denominator>0
        &&quotientFinite(pressure,denominator,rho)&&rho>0
        &&productFinite(rho,D,weighted)&&weighted>0
        &&quotientFinite(weighted,distance,result)&&result>0)return true;
        int ep=0,ed=0,et=0,ex=0,er=0;
        Real m=::frexp(pressure,&ep)*::frexp(D,&ed)
        /(::frexp(T,&et)*::frexp(distance,&ex)*::frexp(Rref,&er));
        int em=0;
        m=::frexp(m,&em);
        int exponent=ep+ed-et-ex-er+em;
        if(exponent>1024 || (exponent==1024 && m>0.99999999999999988898))return false;
        result=exponent < -1074 ? 0 : ::scalbn(m,exponent);
        return finite(result);
    }

    CHMT_HD inline bool sumFinite(Real a,Real b,Real& result) {
        const Real maximum=1.7976931348623157e308;
        if(!finite(a)||!finite(b))return false;
        if(((a>0&&b>0)||(a<0&&b<0)) && (absValue(a)>maximum*.5||absValue(b)>maximum*.5)){
            Real half=a*.5+b*.5;
            if(absValue(half)>maximum*.5){result=a<0?-HUGE_VAL:HUGE_VAL;return false;}
            result=half*2;return true;
        }
        result=a+b;return true;
    }

    CHMT_HD inline bool scaledSum(const Real* values,const Real* factors,Real& result) {
        Real direct=0;
        bool represented=true;
        for(int i=0;i<Ns;++i){
            Real term=values[i];
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
        Real scale=0;
        for(int i=0;i<Ns;++i)scale=maxValue(scale,absValue(values[i]));
        if(scale==0){
            result=0;
            return true;
        }
        Real sum=0;
        for(int i=0;i<Ns;++i){
            Real term=values[i]/scale;
            if(factors&&!productFinite(term,factors[i],term))return false;
            if(!sumFinite(sum,term,sum))return false;
        }
        return productFinite(scale,sum,result);
    }
}
#endif
