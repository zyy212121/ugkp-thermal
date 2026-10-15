#ifndef UGKWP_CONSERVATIVE_REACTOR_H
#define UGKWP_CONSERVATIVE_REACTOR_H
#include "GasJacobian.H"
namespace ugkwp { namespace gasChemistryDetail {
// Equilibrate each equation by its own row norm before partial pivoting.
// Independent kinetic modes can have arbitrarily different rate/time scales;
// a unit slow-mode pivot must not be compared with a 1e100 fast-mode diagonal.


template<class R, int N, bool Refinement=false, bool Complete=!Refinement>
UGKWP_GAS_HD inline bool solveDense(int n, R* a, R* b,bool* evaluated=nullptr) noexcept
{
    using gasThermoDetail::absValue;
    using gasThermoDetail::maxValue;
    if(evaluated)*evaluated=false;
    bool finite=true;
    if ((!Complete && (n < 0 || n > N || !a || !b))
        || (Complete && n>0 && (!a || !b))) return false;
    for (int i = 0; i < n; ++i)
    {
        R rowScale = R(0);
        for (int j = 0; j < n; ++j)
        {
            if (!gasFinite(a[i*N + j])) {finite=false;if(!Complete)return false;}
            rowScale = maxValue(rowScale, absValue(a[i*N + j]));
        }
        if (!(rowScale > R(0)) || !gasFinite(b[i])) {finite=false;if(!Complete)return false;}
        for (int j = 0; j < n; ++j) a[i*N + j] /= rowScale;
        b[i] /= rowScale;
        if (!gasFinite(b[i])) {finite=false;if(!Complete)return false;}
    }

    for (int k = 0; k < n; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (absValue(a[i*N + k]) > absValue(a[pivot*N + k]))
                pivot = i;
        if (!gasFinite(a[pivot*N + k])
            || (Refinement ? absValue(a[pivot*N + k])<=R(8)*std::numeric_limits<R>::epsilon() : a[pivot*N + k]==R(0)))
            {finite=false;if(!Complete)return false;}
        if (pivot != k)
        {
            for (int j = k; j < n; ++j)
            {
                const R value = a[k*N + j];
                a[k*N + j] = a[pivot*N + j];
                a[pivot*N + j] = value;
            }
            const R value = b[k];
            b[k] = b[pivot];
            b[pivot] = value;
        }
        for (int i = k + 1; i < n; ++i)
        {
            const R factor = a[i*N + k]/a[k*N + k];
            for (int j = k + 1; j < n; ++j)
                a[i*N + j] -= factor*a[k*N + j];
            b[i] -= factor*b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i)
    {
        R value = b[i];
        for (int j = i + 1; j < n; ++j) value -= a[i*N + j]*b[j];
        b[i] = value/a[i*N + i];
        if (!gasFinite(b[i])) {finite=false;if(!Complete)return false;}
    }
    if(evaluated)*evaluated=true;
    return finite;
}
template<class R,int N,class TableReal=R> struct ReactorContext {
 const ClosedReactorInput<R,N>& base;
 const SpeciesThermoView<TableReal,N>& thermo;
 const GasMechanismView<TableReal,N>& mechanism;
 const ChemistryControls<R>& controls;
 bool refinement;
 R leftInverse[N*N]; // [coordinate*Ns+species]
 UGKWP_GAS_HD ReactorContext(const ClosedReactorInput<R,N>& b,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& c,bool refine=true) noexcept:base(b),thermo(t),mechanism(m),controls(c),refinement(refine),leftInverse{} {}
 UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF bool prepare(bool* evaluated=nullptr) noexcept {
   if(evaluated)*evaluated=false;
   R q[N*N]={};bool available=false;
   bool finite=refinement?orthogonalBasis<R,N,TableReal,false>(mechanism,q,&available)
     :orthogonalBasis(mechanism,q,&available);
   if(!available)return false;
   int rank=mechanism.independentRank;R upper[N*N]={};
   for(int i=0;i<rank;++i)for(int j=0;j<rank;++j)for(int s=0;s<N;++s)upper[i*N+j]+=q[s*N+i]*mechanism.stoichiometricBasis[s*rank+j];
   for(int s=0;s<N;++s){R a[N*N],b[N]={};for(int k=0;k<N*N;++k)a[k]=upper[k];for(int i=0;i<rank;++i)b[i]=q[s*N+i];
     available=false;
     const bool solved=refinement?solveDense<R,N,false,false>(rank,a,b,&available):solveDense<R,N>(rank,a,b,&available);
     if(!available)return false;
     finite=finite && solved;for(int i=0;i<rank;++i)leftInverse[i*N+s]=b[i];}
   if(evaluated)*evaluated=true;
   return finite;
 }
 UGKWP_GAS_HD bool state(const R* z,R* mass,R& temperature,ThermoStatus* diagnostic=nullptr) const noexcept {
   int rank=mechanism.independentRank;
   if(!thermo.species || !z || !mass || (rank>0 && !mechanism.stoichiometricBasis)
      || (refinement && (rank<0 || rank>N)))return false;
   for(int s=0;s<N;++s){R amount=0;for(int j=0;j<rank;++j)amount+=mechanism.stoichiometricBasis[s*rank+j]*z[j];mass[s]=base.speciesMass[s]+thermo.species[s].molarMass*base.volume*amount;}
   if(refinement)for(int s=0;s<N;++s)if(!gasFinite(mass[s]) || mass[s]<R(0))return false;
   bool available=false;
   const auto status=refinement ? invertMixtureEnergy<true>(mass,base.internalEnergy,thermo,controls.thermo,temperature,&available)
     :invertMixtureEnergy(mass,base.internalEnergy,thermo,controls.thermo,temperature,&available);
   if(diagnostic)*diagnostic=status;
   return status==ThermoStatus::Success || (!refinement && available);
 }

 UGKWP_GAS_HD ChemistryStatus evaluate(const R* z,R* f,R& temperature,R* jacobian,
     R* preparedMasses=nullptr,bool* rhsAvailable=nullptr,bool* jacobianAvailable=nullptr) const noexcept {
   if(rhsAvailable)*rhsAvailable=false;
   ChemistryStatus st;R localMasses[N];R* masses=preparedMasses?preparedMasses:localMasses;
   if(jacobianAvailable)*jacobianAvailable=false;
   if(!state(z,masses,temperature,&st.thermoCode)){st.code=ChemistryCode::TemperatureFailure;return st;}
   ChemistryStatus rates=evaluateState(masses,f,temperature,jacobian,rhsAvailable,jacobianAvailable);
   if(st.thermoCode!=ThermoStatus::Success){st.code=ChemistryCode::TemperatureFailure;return st;}
   return rates;
 }
 UGKWP_GAS_HD ChemistryStatus evaluateState(const R* masses,R* f,R temperature,
     R* jacobian,bool* rhsAvailable=nullptr,bool* jacobianAvailable=nullptr) const noexcept {
   if(rhsAvailable)*rhsAvailable=false;
   ChemistryStatus st;
   if(jacobianAvailable)*jacobianAvailable=false;
   if(!f || !masses || !thermo.species){st.code=ChemistryCode::InvalidComposition;return st;}
   R c[N],production[N],mapped[N]={};
   for(int s=0;s<N;++s){c[s]=masses[s]/(thermo.species[s].molarMass*base.volume);if(refinement && (!gasFinite(c[s]) || c[s]<R(0))){st.code=ChemistryCode::InvalidComposition;return st;}}
   bool available=false;
   const bool rates=refinement?evaluateGasRates<R,N,TableReal,false>(temperature,c,thermo,mechanism,production,nullptr,st,&available)
     :evaluateGasRates(temperature,c,thermo,mechanism,production,nullptr,st,&available);
   if(!rates && (refinement || !available))return st;
   const int rank=mechanism.independentRank;
   for(int i=0;i<rank;++i){for(int s=0;s<N;++s)mapped[i]+=leftInverse[i*N+s]*production[s];
     if(!refinement && !gasFinite(mapped[i]))st.code=ChemistryCode::RateFailure;}
   for(int i=0;i<rank;++i)f[i]=mapped[i];
   if(rhsAvailable)*rhsAvailable=true;
   if(jacobian){const auto derivative=evaluateJacobian(masses,temperature,jacobian,jacobianAvailable);if(st)return derivative;}
   return st;
 }
 UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF ChemistryStatus evaluateJacobian(const R* masses,R temperature,R* jacobian,bool* evaluated=nullptr) const noexcept {
   if(evaluated)*evaluated=false;
   ChemistryStatus st;R c[N],jc[N*N],jt[N],dtdz[N]={};
   if(!masses || (mechanism.independentRank>0 && !jacobian) || !thermo.species){st.code=ChemistryCode::InvalidComposition;return st;}
   for(int s=0;s<N;++s){c[s]=masses[s]/(thermo.species[s].molarMass*base.volume);if(refinement && (!gasFinite(c[s]) || c[s]<R(0))){st.code=ChemistryCode::InvalidComposition;return st;}}
   bool available=false;
   const bool derivatives=refinement?evaluateGasRateDerivatives<R,N,TableReal,false>(temperature,c,thermo,mechanism,jc,jt,st,&available)
     :evaluateGasRateDerivatives(temperature,c,thermo,mechanism,jc,jt,st,&available);
   if(!derivatives && (refinement || !available))return st;
   const int rank=mechanism.independentRank;
   R cv=mixtureHeatCapacity(masses,temperature,thermo);


   if(rank>0 && !refinement && !gasFinite(cv)){
     available=false;const auto caloric=evaluateMixtureHeatCapacity(masses,temperature,thermo,cv,&available);
     if(caloric!=ThermoStatus::Success){st.code=ChemistryCode::TemperatureFailure;st.thermoCode=caloric;}
     if(!available)return st;
   }
   for(int j=0;j<rank;++j)for(int s=0;s<N;++s){
     R energy=speciesE(s,temperature,thermo);
     if(!refinement && !gasFinite(energy)){
       gasThermoDetail::SpeciesValues<R> values;available=false;
       const auto caloric=gasThermoDetail::evaluateSpecies<R,N,TableReal,true,true>(s,temperature,thermo,values,&available);
       if(caloric!=ThermoStatus::Success){st.code=ChemistryCode::TemperatureFailure;st.thermoCode=caloric;}
       if(!available)return st;
       energy=values.e;
     }
     dtdz[j]-=energy*thermo.species[s].molarMass*base.volume*mechanism.stoichiometricBasis[s*rank+j]/cv;
   }
   for(int i=0;i<rank;++i)for(int j=0;j<rank;++j){R v=0;
     for(int s=0;s<N;++s){R ds=jt[s]*dtdz[j];for(int k=0;k<N;++k)ds+=jc[s*N+k]*mechanism.stoichiometricBasis[k*rank+j];v+=leftInverse[i*N+s]*ds;}
     if(!refinement && !gasFinite(v))st.code=ChemistryCode::RateFailure;
     jacobian[i*N+j]=v;
   }
   if(evaluated)*evaluated=true;
   return st;
 }
};
}}
namespace ugkwp {

// basis; masses=baseMass+W*volume*B*z. The fixed formation-inclusive U closes T.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline ChemistryStatus evaluateConservativeReactor
(const ClosedReactorInput<R,N>& base,const R* z,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,R* f,R& temperature,R* jacobian) noexcept {
 ChemistryStatus st;
 gasChemistryDetail::ReactorContext<R,N,TableReal> context(base,t,m,controls,false);
 bool available=false;
 if(!context.prepare(&available)){st.code=ChemistryCode::InvalidModel;if(!available)return st;}
 const auto evaluated=context.evaluate(z,f,temperature,jacobian);
 return st?evaluated:st;
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline ChemistryStatus evaluateConservativeReactor
(const ClosedReactorInput<R,N>& base,const R* z,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,R* f,R& temperature,std::nullptr_t) noexcept {return evaluateConservativeReactor(base,z,t,m,controls,f,temperature,static_cast<R*>(nullptr));}
}
#endif
