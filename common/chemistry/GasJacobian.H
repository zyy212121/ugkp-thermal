#ifndef UGKWP_GAS_JACOBIAN_H
#define UGKWP_GAS_JACOBIAN_H
#include "GasRates.H"
namespace ugkwp {
// Fixed-temperature concentration Jacobian, row-major [species*Ns+variable],
// and independent temperature derivative. The conservative reactor combines
// these with the exact fixed-U thermal feedback derivative.
template<class R,int N> UGKWP_GAS_HD inline bool evaluateGasRateDerivatives
(R T,const R* c,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,R* jacobian,R* thermal,ChemistryStatus& status) noexcept {
 status=gasChemistryDetail::rateInputStatus(T,c,t,m);if(!status)return false;
 if(!jacobian || !thermal){status.code=ChemistryCode::InvalidComposition;return false;}
 using X=gasChemistryDetail::RateDual<R,N+1>;X concentration[N],temperature(T),production[N];
 temperature.derivative[N]=R(1);for(int s=0;s<N;++s){concentration[s]=X(c[s]);concentration[s].derivative[s]=R(1);}
 for(int r=0;r<m.reactionCount;++r){X q;if(!gasChemistryDetail::progressRate(r,temperature,concentration,t,m,q)){status.code=ChemistryCode::RateFailure;status.reaction=r;return false;}
   const auto& a=m.reactions[r];
   for(int j=0;j<a.reactantCount;++j){const auto& v=m.reactants[a.reactantOffset+j];production[v.species]=production[v.species]-X(v.coefficient)*q;}
   for(int j=0;j<a.productCount;++j){const auto& v=m.products[a.productOffset+j];production[v.species]=production[v.species]+X(v.coefficient)*q;}
 }
 for(int s=0;s<N;++s)if(!gasChemistryDetail::rateFinite(production[s])){status.code=ChemistryCode::RateFailure;return false;}
 for(int s=0;s<N;++s){thermal[s]=production[s].derivative[N];for(int j=0;j<N;++j)jacobian[s*N+j]=production[s].derivative[j];}return true;
}
}
#endif
