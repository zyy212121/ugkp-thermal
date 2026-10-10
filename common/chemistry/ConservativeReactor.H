#ifndef UGKWP_CONSERVATIVE_REACTOR_H
#define UGKWP_CONSERVATIVE_REACTOR_H
#include "GasJacobian.H"
namespace ugkwp { namespace gasChemistryDetail {
// Equilibrate each equation by its own row norm before partial pivoting.
// Independent kinetic modes can have arbitrarily different rate/time scales;
// a unit slow-mode pivot must not be compared with a 1e100 fast-mode diagonal.
// The singularity threshold remains relative to each normalized input row.
template<class R, int N>
UGKWP_GAS_HD inline bool solveDense(int n, R* a, R* b) noexcept
{
    using gasThermoDetail::absValue;
    using gasThermoDetail::maxValue;
    if (n < 0 || n > N || !a || !b) return false;
    for (int i = 0; i < n; ++i)
    {
        R rowScale = R(0);
        for (int j = 0; j < n; ++j)
        {
            if (!gasFinite(a[i*N + j])) return false;
            rowScale = maxValue(rowScale, absValue(a[i*N + j]));
        }
        if (!(rowScale > R(0)) || !gasFinite(b[i])) return false;
        for (int j = 0; j < n; ++j) a[i*N + j] /= rowScale;
        b[i] /= rowScale;
        if (!gasFinite(b[i])) return false;
    }

    const R pivotTolerance = R(8)*std::numeric_limits<R>::epsilon();
    for (int k = 0; k < n; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i < n; ++i)
            if (absValue(a[i*N + k]) > absValue(a[pivot*N + k]))
                pivot = i;
        if (!gasFinite(a[pivot*N + k])
            || absValue(a[pivot*N + k]) <= pivotTolerance)
            return false;
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
        if (!gasFinite(b[i])) return false;
    }
    return true;
}
template<class R,int N> struct ReactorContext {
 const ClosedReactorInput<R,N>& base;
 const SpeciesThermoView<R,N>& thermo;
 const GasMechanismView<R,N>& mechanism;
 const ChemistryControls<R>& controls;
 R leftInverse[N*N]; // [coordinate*Ns+species]
 UGKWP_GAS_HD ReactorContext(const ClosedReactorInput<R,N>& b,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,const ChemistryControls<R>& c) noexcept:base(b),thermo(t),mechanism(m),controls(c),leftInverse{} {}
 UGKWP_GAS_HD bool prepare() noexcept {
   R q[N*N]={};if(!orthogonalBasis(mechanism,q))return false;
   int rank=mechanism.independentRank;R upper[N*N]={};
   for(int i=0;i<rank;++i)for(int j=0;j<rank;++j)for(int s=0;s<N;++s)upper[i*N+j]+=q[s*N+i]*mechanism.stoichiometricBasis[s*rank+j];
   for(int s=0;s<N;++s){R a[N*N],b[N]={};for(int k=0;k<N*N;++k)a[k]=upper[k];for(int i=0;i<rank;++i)b[i]=q[s*N+i];if(!solveDense<R,N>(rank,a,b))return false;for(int i=0;i<rank;++i)leftInverse[i*N+s]=b[i];}return true;
 }
 UGKWP_GAS_HD bool state(const R* z,R* mass,R& temperature) const noexcept {
   int rank=mechanism.independentRank;
   for(int s=0;s<N;++s){R amount=0;for(int j=0;j<rank;++j)amount+=mechanism.stoichiometricBasis[s*rank+j]*z[j];mass[s]=base.speciesMass[s]+thermo.species[s].molarMass*base.volume*amount;if(!gasFinite(mass[s]) || mass[s]<R(0))return false;}
   return invertMixtureEnergy(mass,base.internalEnergy,thermo,controls.thermo,temperature)==ThermoStatus::Success;
 }
 UGKWP_GAS_HD ChemistryStatus evaluate(const R* z,R* f,R& temperature,R* jacobian) const noexcept {
   ChemistryStatus st;R masses[N],c[N],production[N];
   if(!state(z,masses,temperature)){st.code=ChemistryCode::TemperatureFailure;return st;}
   for(int s=0;s<N;++s)c[s]=masses[s]/(thermo.species[s].molarMass*base.volume);
   if(!evaluateGasRates(temperature,c,thermo,mechanism,production,nullptr,st))return st;
   int rank=mechanism.independentRank;
   for(int i=0;i<rank;++i){f[i]=0;for(int s=0;s<N;++s)f[i]+=leftInverse[i*N+s]*production[s];}
   if(jacobian){R jc[N*N],jt[N],dtdz[N]={};if(!evaluateGasRateDerivatives(temperature,c,thermo,mechanism,jc,jt,st))return st;
     const R cv=mixtureHeatCapacity(masses,temperature,thermo);
     for(int j=0;j<rank;++j)for(int s=0;s<N;++s)dtdz[j]-=speciesE(s,temperature,thermo)*thermo.species[s].molarMass*base.volume*mechanism.stoichiometricBasis[s*rank+j]/cv;
     for(int i=0;i<rank;++i)for(int j=0;j<rank;++j){R v=0;for(int s=0;s<N;++s){R ds=jt[s]*dtdz[j];for(int k=0;k<N;++k)ds+=jc[s*N+k]*mechanism.stoichiometricBasis[k*rank+j];v+=leftInverse[i*N+s]*ds;}jacobian[i*N+j]=v;}
   }
   return st;
 }
};
}}
namespace ugkwp {
// Coordinates z are mol/m^3 increments in the validated dimensionless MOLAR
// basis; masses=baseMass+W*volume*B*z. The fixed formation-inclusive U closes T.
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus evaluateConservativeReactor
(const ClosedReactorInput<R,N>& base,const R* z,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,const ChemistryControls<R>& controls,R* f,R& temperature,R* jacobian) noexcept {
 ChemistryStatus st=validateGasMechanismView(t,m);if(!st)return st;
 gasChemistryDetail::ReactorContext<R,N> context(base,t,m,controls);
 if(!context.prepare()){st.code=ChemistryCode::InvalidModel;return st;}
 return context.evaluate(z,f,temperature,jacobian);
}
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus evaluateConservativeReactor
(const ClosedReactorInput<R,N>& base,const R* z,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,const ChemistryControls<R>& controls,R* f,R& temperature,std::nullptr_t) noexcept {return evaluateConservativeReactor(base,z,t,m,controls,f,temperature,static_cast<R*>(nullptr));}
}
#endif
