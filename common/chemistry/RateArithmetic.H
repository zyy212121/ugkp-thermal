#ifndef UGKWP_RATE_ARITHMETIC_H
#define UGKWP_RATE_ARITHMETIC_H
#include "../gasTransport/MixtureThermo.H"
namespace ugkwp { namespace gasChemistryDetail {
// Forward-mode directional arithmetic provides exact concentration/collider/
// falloff/temperature derivatives, including zero concentrations. Bounded stack
// storage and ordinary C++14 operators are shared by host and CUDA compilation.
template<class R,int D> struct RateDual {
 R value; R derivative[D];
 UGKWP_GAS_HD RateDual(R x=R(0)) noexcept:value(x),derivative{} {}
};
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> operator+(const RateDual<R,D>& a,const RateDual<R,D>& b) noexcept {RateDual<R,D> c(a.value+b.value);for(int i=0;i<D;++i)c.derivative[i]=a.derivative[i]+b.derivative[i];return c;}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> operator-(const RateDual<R,D>& a,const RateDual<R,D>& b) noexcept {RateDual<R,D> c(a.value-b.value);for(int i=0;i<D;++i)c.derivative[i]=a.derivative[i]-b.derivative[i];return c;}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> operator*(const RateDual<R,D>& a,const RateDual<R,D>& b) noexcept {RateDual<R,D> c(a.value*b.value);for(int i=0;i<D;++i)c.derivative[i]=a.derivative[i]*b.value+a.value*b.derivative[i];return c;}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> operator/(const RateDual<R,D>& a,const RateDual<R,D>& b) noexcept {RateDual<R,D> c(a.value/b.value);for(int i=0;i<D;++i)c.derivative[i]=(a.derivative[i]-c.value*b.derivative[i])/b.value;return c;}
template<class R> UGKWP_GAS_HD inline R rateValue(R a) noexcept {return a;}
template<class R,int D> UGKWP_GAS_HD inline R rateValue(const RateDual<R,D>& a) noexcept {return a.value;}
template<class R> UGKWP_GAS_HD inline R rateExp(R a) noexcept {return ::exp(a);}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> rateExp(const RateDual<R,D>& a) noexcept {RateDual<R,D> c(::exp(a.value));for(int i=0;i<D;++i)c.derivative[i]=c.value*a.derivative[i];return c;}
template<class R> UGKWP_GAS_HD inline R rateLog(R a) noexcept {return ::log(a);}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> rateLog(const RateDual<R,D>& a) noexcept {RateDual<R,D> c(::log(a.value));for(int i=0;i<D;++i)c.derivative[i]=a.derivative[i]/a.value;return c;}
template<class R> UGKWP_GAS_HD inline R rateExpm1(R a) noexcept {return ::expm1(a);}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> rateExpm1(const RateDual<R,D>& a) noexcept {RateDual<R,D> c(::expm1(a.value));R d=::exp(a.value);for(int i=0;i<D;++i)c.derivative[i]=d*a.derivative[i];return c;}
template<class R> UGKWP_GAS_HD inline R ratePow(R a,R b) noexcept {return ::pow(a,b);}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> ratePow(const RateDual<R,D>& a,R b) noexcept {RateDual<R,D> c(::pow(a.value,b));R d=b==R(0)?R(0):(b==R(1)?R(1):b*::pow(a.value,b-R(1)));for(int i=0;i<D;++i)c.derivative[i]=a.derivative[i]==R(0)?R(0):d*a.derivative[i];return c;}
// Evaluate a*b*exp(exponent) without overflowing an intermediate equilibrium
// factor. At either exact-zero factor, differentiate the product analytically;
// this retains finite nonzero collider/zero-product derivatives without floors.
template<class R> UGKWP_GAS_HD inline R rateProductExp(R a,R b,R exponent) noexcept {
 return a==R(0) || b==R(0)?R(0):R(::exp(::log(a)+::log(b)+exponent));
}
template<class R,int D> UGKWP_GAS_HD RateDual<R,D> rateProductExp(const RateDual<R,D>& a,const RateDual<R,D>& b,const RateDual<R,D>& exponent) noexcept {
 RateDual<R,D> out(rateProductExp(a.value,b.value,exponent.value));
 for(int i=0;i<D;++i){R da=0,db=0;
   if(a.derivative[i]!=R(0) && b.value!=R(0)){
     const R sign=a.derivative[i]>R(0)?R(1):R(-1);
     da=sign*::exp(::log(sign*a.derivative[i])+::log(b.value)+exponent.value);}
   if(b.derivative[i]!=R(0) && a.value!=R(0)){
     const R sign=b.derivative[i]>R(0)?R(1):R(-1);
     db=sign*::exp(::log(sign*b.derivative[i])+::log(a.value)+exponent.value);}
   out.derivative[i]=da+db+(out.value==R(0)?R(0):out.value*exponent.derivative[i]);
 }return out;
}
template<class R> UGKWP_GAS_HD inline bool rateFinite(R a) noexcept{return gasFinite(a);}
template<class R,int D> UGKWP_GAS_HD inline bool rateFinite(const RateDual<R,D>& a) noexcept {if(!gasFinite(a.value))return false;for(int i=0;i<D;++i)if(!gasFinite(a.derivative[i]))return false;return true;}
template<class R> UGKWP_GAS_HD inline R gibbsNumber(R value,R) noexcept{return value;}
template<class R,int D> UGKWP_GAS_HD inline RateDual<R,D> gibbsNumber(R value,R thermalDerivative) noexcept {RateDual<R,D> out(value);out.derivative[D-1]=thermalDerivative;return out;}
template<class Number,class R> struct GibbsNumber {UGKWP_GAS_HD static Number make(R v,R) noexcept{return Number(v);}};
template<class R,int D> struct GibbsNumber<RateDual<R,D>,R> {UGKWP_GAS_HD static RateDual<R,D> make(R v,R dt) noexcept {RateDual<R,D> out(v);out.derivative[D-1]=dt;return out;}};
}}
#endif
