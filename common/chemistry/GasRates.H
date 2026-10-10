#ifndef UGKWP_GAS_RATES_H
#define UGKWP_GAS_RATES_H
#include "MechanismValidation.H"
#include "RateArithmetic.H"
#include <cstddef>
namespace ugkwp {
namespace gasChemistryDetail {
template<class X,class R> UGKWP_GAS_HD inline X arrhenius(const GasArrheniusData<R>& a,const X& T) noexcept {
 using C=decltype(rateValue(T));
 if(a.preExponential==R(0))return X(0);
 return rateExp(X(::log(C(a.preExponential)))+X(a.temperatureExponent)*rateLog(T)-X(C(a.activationEnergy)/universalGasConstant<C>())/T);
}
template<class X,class R> UGKWP_GAS_HD inline X concentrationProduct(const GasStoichTerm<R>* terms,int offset,int count,const X* c) noexcept {
 X product(1);for(int j=0;j<count;++j){const auto& a=terms[offset+j];product=product*ratePow(c[a.species],decltype(rateValue(c[0]))(a.coefficient));}return product;
}
template<class X,class R,int N> UGKWP_GAS_HD inline bool progressRate
(int r,const X& T,const X* c,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,X& q) noexcept {
 using C=decltype(rateValue(T));
 const auto& a=m.reactions[r];X k=arrhenius(a.highRate,T);
 if(a.type!=GasReactionType::Elementary) {
   X collider(0);
   // Sum positive efficiencies directly, avoiding subtraction cancellation when
   // an excluded explicit collider has default efficiency zero.
   for(int s=0;s<N;++s){C efficiency=C(a.defaultEfficiency);for(int j=0;j<a.efficiencyCount;++j)if(m.efficiencies[a.efficiencyOffset+j].species==s)efficiency=m.efficiencies[a.efficiencyOffset+j].efficiency;collider=collider+X(efficiency)*c[s];}
   if(a.type==GasReactionType::ThirdBody)k=k*collider;
   else {
     const X low=arrhenius(a.lowRate,T);
     if(rateValue(k)==R(0)){q=X(0);return rateFinite(low);}
     const X reduced=low*collider/k;
     X broadening(1);
     if(a.type==GasReactionType::Troe){
       X center=X(C(1)-C(a.troe.alpha))*rateExp(X(-1)*T/X(a.troe.T3))+X(a.troe.alpha)*rateExp(X(-1)*T/X(a.troe.T1));
       if(a.troe.hasT2)center=center+rateExp(X(-a.troe.T2)/T);
       if(!(rateValue(center)>R(0)))return false;
       const X logCenter=rateLog(center)/X(::log(C(10)));
       X shape;
       if(rateValue(reduced)>R(0)){
         const X cT=X(-.4)-X(.67)*logCenter;
         const X nT=X(.75)-X(1.27)*logCenter;
         const X shifted=rateLog(reduced)/X(::log(C(10)))+cT;
         const X denominator=nT-X(.14)*shifted;
         // Algebraic form avoids the removable infinity when denominator=0.
         shape=denominator*denominator/(denominator*denominator+shifted*shifted);
       } else shape=X(C(.14)*C(.14)/(C(1)+C(.14)*C(.14)));
       broadening=rateExp(X(::log(C(10)))*logCenter*shape);
     }
     k=low*collider/(X(1)+reduced)*broadening;
   }
 }
 const X forward=k*concentrationProduct(m.reactants,a.reactantOffset,a.reactantCount,c);
 X backward(0);
 if(a.reversible){
   X deltaG(0);C deltaNu=0;
   for(int side=0;side<2;++side){const auto* terms=side?m.products:m.reactants;int offset=side?a.productOffset:a.reactantOffset;int count=side?a.productCount:a.reactantCount;
     for(int j=0;j<count;++j){const auto& term=terms[offset+j];const C sign=side?term.coefficient:-term.coefficient;const C Tvalue=rateValue(T);const C W=t.species[term.species].molarMass;
       const C g=speciesGibbsStandard(term.species,Tvalue,t)*W;
       const C dg=-speciesEntropyStandard(term.species,Tvalue,t)*W;
       deltaG=deltaG+X(sign)*GibbsNumber<X,C>::make(g,dg);deltaNu+=sign;}
   }
   const X logKc=X(-1)*deltaG/(X(universalGasConstant<C>())*T)+X(deltaNu)*rateLog(X(C(m.referencePressure)/universalGasConstant<C>())/T);
   const X reverseFactor=concentrationProduct(m.products,a.productOffset,a.productCount,c);
   // This also handles exact-zero products and colliders with finite limiting
   // derivatives, without forming an overflowing reciprocal Kc.
   backward=rateProductExp(k,reverseFactor,X(-1)*logKc);
 }
 q=forward-backward;
 if(rateValue(forward)>R(0) && rateValue(backward)>R(0)) {
   const C ratio=rateValue(backward)/rateValue(forward);
   if(ratio>R(.5) && ratio<R(2))q=X(-1)*forward*rateExpm1(rateLog(backward)-rateLog(forward));
 }
 return rateFinite(q);
}
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus rateInputStatus
(R T,const R* c,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m) noexcept {
 ChemistryStatus st=validateGasMechanismView(t,m);if(!st)return st;
 if(!c || !gasFinite(T)){st.code=ChemistryCode::NonFiniteInput;return st;}
 for(int s=0;s<N;++s){if(!gasFinite(c[s]) || c[s]<R(0)){st.code=ChemistryCode::InvalidComposition;return st;}
   if(!gasFinite(speciesE(s,T,t))){st.code=ChemistryCode::TemperatureFailure;return st;}}
 return st;
}
}
// Production [mol/(m^3 s)] and optional net progress [mol/(m^3 s)]. All caller
// outputs are transactional, including optional progress arrays with runtime Nr.
template<class R,int N> UGKWP_GAS_HD inline bool evaluateGasRates
(R T,const R* c,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,R* production,R* progress,ChemistryStatus& status) noexcept {
 status=gasChemistryDetail::rateInputStatus(T,c,t,m);if(!status)return false;
 if(!production){status.code=ChemistryCode::InvalidComposition;return false;}
 R local[N]={};
 for(int r=0;r<m.reactionCount;++r){R q;
   if(!gasChemistryDetail::progressRate(r,T,c,t,m,q)){status.code=ChemistryCode::RateFailure;status.reaction=r;return false;}
   const auto& a=m.reactions[r];
   for(int j=0;j<a.reactantCount;++j){const auto& v=m.reactants[a.reactantOffset+j];local[v.species]-=v.coefficient*q;}
   for(int j=0;j<a.productCount;++j){const auto& v=m.products[a.productOffset+j];local[v.species]+=v.coefficient*q;}
 }
 for(int s=0;s<N;++s)if(!gasFinite(local[s])){status.code=ChemistryCode::RateFailure;return false;}
 if(progress)for(int r=0;r<m.reactionCount;++r)gasChemistryDetail::progressRate(r,T,c,t,m,progress[r]);
 for(int s=0;s<N;++s)production[s]=local[s];
 return true;
}
template<class R,int N> UGKWP_GAS_HD inline bool evaluateGasRates
(R T,const R* c,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,R* production,std::nullptr_t,ChemistryStatus& status) noexcept {return evaluateGasRates(T,c,t,m,production,static_cast<R*>(nullptr),status);}
}
#endif
