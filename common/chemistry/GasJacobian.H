#ifndef UGKWP_GAS_JACOBIAN_H
#define UGKWP_GAS_JACOBIAN_H
#include "GasRates.H"
// Bound device matrix/stage scratch lifetimes; ordinary host inlining stays intact.
#if defined(__CUDA_ARCH__)
#define UGKWP_CHEMISTRY_DEVICE_LEAF __noinline__
#else
#define UGKWP_CHEMISTRY_DEVICE_LEAF inline
#endif
namespace ugkwp {
// Fixed-temperature concentration Jacobian, row-major [species*Ns+variable],
// and independent temperature derivative. The conservative reactor combines
// these with the exact fixed-U thermal feedback derivative.
template<class R,int N,class TableReal,bool Complete=true> UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF bool evaluateGasRateDerivatives
(R T,const R* c,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,R* jacobian,R* thermal,ChemistryStatus& status,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 status=gasChemistryDetail::rateInputStatus<R,N,TableReal,Complete>(T,c,t,m);if(!status)return false;
 if(!jacobian || !thermal){status.code=ChemistryCode::InvalidComposition;return false;}
 using X=gasChemistryDetail::RateDual<R,N+1>;X concentration[N],temperature(T),production[N];
 temperature.derivative[N]=R(1);if(c)for(int s=0;s<N;++s){concentration[s]=X(c[s]);concentration[s].derivative[s]=R(1);}
 for(int r=0;r<m.reactionCount;++r){X q;bool available=false;
   if(!gasChemistryDetail::progressRate<X,TableReal,N,Complete>(r,temperature,c?concentration:nullptr,t,m,q,&available)){
     if(status){status.code=ChemistryCode::RateFailure;status.reaction=r;}
     if(!Complete || !available)return false;
   }
   const auto& a=m.reactions[r];
   for(int j=0;j<a.reactantCount;++j){const auto& v=m.reactants[a.reactantOffset+j];production[v.species]=production[v.species]-X(v.coefficient)*q;}
   for(int j=0;j<a.productCount;++j){const auto& v=m.products[a.productOffset+j];production[v.species]=production[v.species]+X(v.coefficient)*q;}
 }
 for(int s=0;s<N;++s)if(!gasChemistryDetail::rateFinite(production[s])){status.code=ChemistryCode::RateFailure;if(!Complete)return false;}
 for(int s=0;s<N;++s){thermal[s]=production[s].derivative[N];for(int j=0;j<N;++j)jacobian[s*N+j]=production[s].derivative[j];}
 if(evaluated)*evaluated=true;
 return bool(status);
}
}
#endif
