#ifndef UGKWP_MIXTURE_THERMO_H
#define UGKWP_MIXTURE_THERMO_H

#include "GasModel.H"
#include <cmath>
#include <limits>

namespace ugkwp
{

// These utilities use no application scalar aliases, allocation, exceptions, or
// I/O and compile identically in a host test and a CUDA translation unit.
template<class Real>
UGKWP_GAS_HD inline bool gasFinite(const Real x) noexcept
{
    return x == x && x <= (std::numeric_limits<Real>::max)()
        && x >= -(std::numeric_limits<Real>::max)();
}

namespace gasThermoDetail
{
template<class Real> UGKWP_GAS_HD inline Real absValue(Real x) noexcept
{ return x < Real(0) ? -x : x; }
template<class Real> UGKWP_GAS_HD inline Real maxValue(Real a, Real b) noexcept
{ return a > b ? a : b; }
template<class Real> UGKWP_GAS_HD inline Real minValue(Real a, Real b) noexcept
{ return a < b ? a : b; }
template<class Real> UGKWP_GAS_HD inline Real invalidValue() noexcept
{ return std::numeric_limits<Real>::quiet_NaN(); }

template<class Real, int Ns>
UGKWP_GAS_HD inline bool validMetadata
(
    int species, const SpeciesThermoView<Real,Ns>& view
) noexcept
{
    if (species < 0 || species >= Ns || !view.species || !view.coefficients)
        return false;
    const SpeciesThermoData<Real>& data = view.species[species];
    int count = 0;
    if (data.model == SpeciesThermoModel::LinearCp) count = 3;
    else if (data.model == SpeciesThermoModel::NASA7) count = 14;
    else return false;
    // Check subtraction before adding the offset, avoiding signed overflow even
    // for malformed input with coefficientOffset==INT_MAX.
    if (view.coefficientCount < count || data.coefficientOffset < 0
        || data.coefficientOffset > view.coefficientCount-count)
        return false;
    if (!gasFinite(data.molarMass) || !(data.molarMass > Real(0))
        || !gasFinite(data.minTemperature) || !(data.minTemperature > Real(0))
        || !gasFinite(data.maxTemperature)
        || !(data.maxTemperature > data.minTemperature)
        || !gasFinite(data.referencePressure) || !(data.referencePressure > Real(0)))
        return false;
    if (data.model == SpeciesThermoModel::NASA7)
    {
        if (!gasFinite(data.midTemperature)
            || !(data.midTemperature > data.minTemperature)
            || !(data.midTemperature < data.maxTemperature)) return false;
    }
    else if (data.hasEntropyReference
        && (!gasFinite(data.referenceTemperature) || !(data.referenceTemperature > Real(0))
            || !gasFinite(data.referenceEntropy))) return false;
    for (int j=0; j<count; ++j)
        if (!gasFinite(view.coefficients[data.coefficientOffset+j])) return false;
    return true;
}

template<class Real>
struct SpeciesValues
{
    Real cp, cv, h, e, entropy;
};

template<class Real, int Ns>
UGKWP_GAS_HD inline ThermoStatus evaluateSpecies
(
    int species, Real temperature, const SpeciesThermoView<Real,Ns>& view,
    SpeciesValues<Real>& output
) noexcept
{
    if (!validMetadata(species,view)) return ThermoStatus::InvalidModel;
    if (!gasFinite(temperature)) return ThermoStatus::NonFiniteInput;
    const SpeciesThermoData<Real>& data = view.species[species];
    if (temperature < data.minTemperature || temperature > data.maxTemperature)
        return ThermoStatus::TemperatureOutOfRange;
    const Real* c = view.coefficients + data.coefficientOffset;
    const Real R = universalGasConstant<Real>()/data.molarMass;
    SpeciesValues<Real> local;
    if (data.model == SpeciesThermoModel::LinearCp)
    {
        local.cp = c[0] + c[1]*temperature;
        local.h = c[2] + temperature*(c[0] + Real(0.5)*c[1]*temperature);
        local.entropy = data.hasEntropyReference
            ? data.referenceEntropy + c[0]*::log(temperature/data.referenceTemperature)
                + c[1]*(temperature-data.referenceTemperature)
            : invalidValue<Real>();
    }
    else
    {
        if (temperature > data.midTemperature) c += 7;
        local.cp = R*(c[0]+temperature*(c[1]+temperature*(c[2]+temperature*(c[3]+temperature*c[4]))));
        local.h = R*(c[5]+temperature*(c[0]+temperature*(c[1]/Real(2)
            +temperature*(c[2]/Real(3)+temperature*(c[3]/Real(4)+temperature*c[4]/Real(5))))));
        local.entropy = R*(c[0]*::log(temperature)+c[6]+temperature*(c[1]
            +temperature*(c[2]/Real(2)+temperature*(c[3]/Real(3)+temperature*c[4]/Real(4)))));
    }
    local.cv = local.cp-R;
    local.e = local.h-R*temperature;
    if (!gasFinite(local.cp) || !gasFinite(local.cv) || !gasFinite(local.h)
        || !gasFinite(local.e)) return ThermoStatus::NonFiniteInput;
    if (!(local.cv > Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    output = local;
    return ThermoStatus::Success;
}

template<int Degree, class Real>
UGKWP_GAS_HD inline Real polynomial(const Real* c, Real x) noexcept
{
    Real result=c[Degree];
    for(int j=Degree-1;j>=0;--j) result=result*x+c[j];
    return result;
}

// Isolate real polynomial roots on a bounded interval by splitting at roots of
// the derivative. Degree is compile-time bounded (at most three), so CUDA does
// not require dynamic allocation or runtime-recursive call stacks.
template<int Degree>
struct PolynomialRoots
{
    template<class Real>
    UGKWP_GAS_HD static int find(const Real* c, Real lower, Real upper, Real* roots) noexcept
    {
        Real derivative[Degree];
        for(int j=0;j<Degree;++j) derivative[j]=Real(j+1)*c[j+1];
        Real critical[Degree-1];
        const int count=PolynomialRoots<Degree-1>::find(derivative,lower,upper,critical);
        int rootCount=0;
        Real left=lower;
        Real leftValue=polynomial<Degree>(c,left);
        for(int interval=0;interval<=count;++interval)
        {
            const Real right=interval<count?critical[interval]:upper;
            const Real rightValue=polynomial<Degree>(c,right);
            if (left>lower && leftValue==Real(0) && rootCount<Degree)
                roots[rootCount++]=left;
            if ((leftValue<Real(0) && rightValue>Real(0))
                || (leftValue>Real(0) && rightValue<Real(0)))
            {
                Real lo=left,hi=right,flo=leftValue;
                for(int i=0;i<96;++i)
                {
                    const Real mid=lo+Real(0.5)*(hi-lo);
                    if(mid==lo || mid==hi) break;
                    const Real fm=polynomial<Degree>(c,mid);
                    if(fm==Real(0)) { lo=hi=mid; break; }
                    if((flo<Real(0))==(fm<Real(0))) { lo=mid; flo=fm; }
                    else hi=mid;
                }
                if(rootCount<Degree) roots[rootCount++]=lo+Real(0.5)*(hi-lo);
            }
            left=right;
            leftValue=rightValue;
        }
        return rootCount;
    }
};

template<>
struct PolynomialRoots<1>
{
    template<class Real>
    UGKWP_GAS_HD static int find(const Real* c,Real lower,Real upper,Real* roots) noexcept
    {
        if(c[1]==Real(0)) return 0;
        const Real root=-c[0]/c[1];
        if(root>lower && root<upper && gasFinite(root)) { roots[0]=root; return 1; }
        return 0;
    }
};

template<class Real>
UGKWP_GAS_HD inline bool nasaPositiveCv(const Real* c,Real lo,Real hi) noexcept
{
    // Test the exact quartic minimum: both endpoints and every interior root
    // of its cubic derivative. Endpoint-only sampling misses negative cv wells.
    Real derivative[4]={c[1],Real(2)*c[2],Real(3)*c[3],Real(4)*c[4]};
    Real extrema[3];
    const int count=PolynomialRoots<3>::find(derivative,lo,hi,extrema);
    const Real lowValue=polynomial<4>(c,lo),highValue=polynomial<4>(c,hi);
    if(!gasFinite(lowValue) || !gasFinite(highValue) || !(lowValue>Real(1)) || !(highValue>Real(1))) return false;
    for(int i=0;i<count;++i)
        {
            const Real value=polynomial<4>(c,extrema[i]);
            if(!gasFinite(value) || !(value>Real(1))) return false;
        }
    return true;
}

} // namespace gasThermoDetail

// Intended at host model-load time. Full-range cv validation does not refit or
// smooth NASA source coefficients. Caloric evaluation remains available for a
// LinearCp table without entropy; Gibbs-based reverse rates must require entropy.
template<class Real, int Ns>
UGKWP_GAS_HD inline ThermoStatus validateSpeciesThermoView
(
    const SpeciesThermoView<Real,Ns>& view
) noexcept
{
    using namespace gasThermoDetail;
    if(view.elementCount<0 || (view.elementCount>0 && !view.elementComposition))
        return ThermoStatus::InvalidModel;
    Real commonLower=Real(0),commonUpper=(std::numeric_limits<Real>::max)();
    for(int species=0;species<Ns;++species)
    {
        if(!validMetadata(species,view)) return ThermoStatus::InvalidModel;
        const SpeciesThermoData<Real>& data=view.species[species];
        commonLower=maxValue(commonLower,data.minTemperature);
        commonUpper=minValue(commonUpper,data.maxTemperature);
        const Real* c=view.coefficients+data.coefficientOffset;
        if(data.model==SpeciesThermoModel::LinearCp)
        {
            const Real R=universalGasConstant<Real>()/data.molarMass;
            const Real lowCp=c[0]+c[1]*data.minTemperature;
            const Real highCp=c[0]+c[1]*data.maxTemperature;
            if(!gasFinite(lowCp) || !gasFinite(highCp)) return ThermoStatus::NonFiniteInput;
            if(!(lowCp>R) || !(highCp>R))
                return ThermoStatus::NonPositiveHeatCapacity;
        }
        else if(!nasaPositiveCv(c,data.minTemperature,data.midTemperature)
            || !nasaPositiveCv(c+7,data.midTemperature,data.maxTemperature))
            return ThermoStatus::NonPositiveHeatCapacity;
        SpeciesValues<Real> edgeValues;
        ThermoStatus edgeStatus=evaluateSpecies(species,data.minTemperature,view,edgeValues);
        if(edgeStatus!=ThermoStatus::Success) return edgeStatus;
        edgeStatus=evaluateSpecies(species,data.maxTemperature,view,edgeValues);
        if(edgeStatus!=ThermoStatus::Success) return edgeStatus;
        for(int element=0;element<view.elementCount;++element)
        {
            const Real count=view.elementComposition[species*view.elementCount+element];
            if(!gasFinite(count) || count<Real(0)) return ThermoStatus::InvalidModel;
        }
    }
    return commonUpper>commonLower?ThermoStatus::Success:ThermoStatus::InvalidModel;
}

#define UGKWP_DEFINE_SPECIES_PROPERTY(Name, Member) \
template<class Real, int Ns> \
UGKWP_GAS_HD inline Real Name \
(int species,Real temperature,const SpeciesThermoView<Real,Ns>& view) noexcept \
{ \
    gasThermoDetail::SpeciesValues<Real> values; \
    return gasThermoDetail::evaluateSpecies(species,temperature,view,values)==ThermoStatus::Success \
        && gasFinite(values.Member) ? values.Member : gasThermoDetail::invalidValue<Real>(); \
}
UGKWP_DEFINE_SPECIES_PROPERTY(speciesCp,cp)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesCv,cv)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesH,h)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesE,e)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesEntropyStandard,entropy)
#undef UGKWP_DEFINE_SPECIES_PROPERTY

template<class Real,int Ns>
UGKWP_GAS_HD inline Real speciesGibbsStandard
(int species,Real temperature,const SpeciesThermoView<Real,Ns>& view) noexcept
{
    gasThermoDetail::SpeciesValues<Real> values;
    if(gasThermoDetail::evaluateSpecies(species,temperature,view,values)!=ThermoStatus::Success)
        return gasThermoDetail::invalidValue<Real>();
    const Real gibbs=values.h-temperature*values.entropy;
    return gasFinite(gibbs)?gibbs:gasThermoDetail::invalidValue<Real>();
}

// The arrays contain masses [kg], not mass fractions or molar concentrations.
// Supplying mass fractions is valid only when intentionally evaluating one kg.
template<class Real,int Ns>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureEnergy
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<Real,Ns>& view,
    Real& energy,Real& heatCapacity
) noexcept
{
    if(!speciesMass) return ThermoStatus::InvalidComposition;
    Real totalMass=Real(0),localEnergy=Real(0),localCapacity=Real(0);
    for(int s=0;s<Ns;++s)
    {
        if(!gasFinite(speciesMass[s]) || speciesMass[s]<Real(0))
            return ThermoStatus::InvalidComposition;
        gasThermoDetail::SpeciesValues<Real> values;
        const ThermoStatus status=gasThermoDetail::evaluateSpecies(s,temperature,view,values);
        if(status!=ThermoStatus::Success) return status;
        totalMass+=speciesMass[s];
        localEnergy+=speciesMass[s]*values.e;
        localCapacity+=speciesMass[s]*values.cv;
    }
    if(!gasFinite(totalMass) || !(totalMass>Real(0))) return ThermoStatus::InvalidComposition;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) return ThermoStatus::NonFiniteInput;
    if(!(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;
    heatCapacity=localCapacity;
    return ThermoStatus::Success;
}

template<class Real,int Ns>
UGKWP_GAS_HD inline Real mixtureEnergy
(const Real* speciesMass,Real temperature,const SpeciesThermoView<Real,Ns>& view) noexcept
{
    Real energy,capacity;
    return evaluateMixtureEnergy(speciesMass,temperature,view,energy,capacity)==ThermoStatus::Success
        ? energy : gasThermoDetail::invalidValue<Real>();
}

template<class Real,int Ns>
UGKWP_GAS_HD inline Real mixtureHeatCapacity
(const Real* speciesMass,Real temperature,const SpeciesThermoView<Real,Ns>& view) noexcept
{
    Real energy,capacity;
    return evaluateMixtureEnergy(speciesMass,temperature,view,energy,capacity)==ThermoStatus::Success
        ? capacity : gasThermoDetail::invalidValue<Real>();
}

// Mass-specific mixture R. Input may contain masses or mass fractions; total
// mass is explicitly divided out. Invalid/zero mass has no meaningful R.
template<class Real,int Ns>
UGKWP_GAS_HD inline Real mixtureGasConstant
(const Real* speciesMass,const SpeciesThermoView<Real,Ns>& view) noexcept
{
    if(!speciesMass || validateSpeciesThermoView(view)!=ThermoStatus::Success)
        return gasThermoDetail::invalidValue<Real>();
    Real totalMass=Real(0),weightedConstant=Real(0);
    for(int species=0;species<Ns;++species)
    {
        if(!gasFinite(speciesMass[species]) || speciesMass[species]<Real(0))
            return gasThermoDetail::invalidValue<Real>();
        totalMass+=speciesMass[species];
        weightedConstant+=speciesMass[species]*universalGasConstant<Real>()/view.species[species].molarMass;
    }
    if(!gasFinite(totalMass) || !(totalMass>Real(0)) || !gasFinite(weightedConstant))
        return gasThermoDetail::invalidValue<Real>();
    const Real result=weightedConstant/totalMass;
    return gasFinite(result) && result>Real(0)?result:gasThermoDetail::invalidValue<Real>();
}

// p/T/Y initialization has one common EOS owner. Y must already be admissible;
// this routine never repairs species or silently accepts arbitrary mass units.
// All species (including zero-mass products) must share the supplied T range.
template<class Real,int Ns>
UGKWP_GAS_HD inline ThermoStatus mixtureDensityFromPressureTemperature
(
    Real pressure,Real temperature,const Real* massFractions,
    const SpeciesThermoView<Real,Ns>& view,Real& density,
    Real compositionTolerance=Real(64)*std::numeric_limits<Real>::epsilon()
) noexcept
{
    if(!gasFinite(pressure) || !gasFinite(temperature)) return ThermoStatus::NonFiniteInput;
    if(!(pressure>Real(0)) || !(temperature>Real(0))) return ThermoStatus::InvalidComposition;
    if(!gasFinite(compositionTolerance) || compositionTolerance<Real(0)) return ThermoStatus::InvalidControls;
    if(!massFractions) return ThermoStatus::InvalidComposition;
    Real total=Real(0);
    for(int species=0;species<Ns;++species)
    {
        if(!gasFinite(massFractions[species]) || massFractions[species]<Real(0)
            || massFractions[species]>Real(1)) return ThermoStatus::InvalidComposition;
        total+=massFractions[species];
    }
    if(!gasFinite(total) || gasThermoDetail::absValue(total-Real(1))>compositionTolerance)
        return ThermoStatus::InvalidComposition;
    Real energy,capacity;
    const ThermoStatus status=evaluateMixtureEnergy(massFractions,temperature,view,energy,capacity);
    if(status!=ThermoStatus::Success) return status;
    const Real gasConstant=mixtureGasConstant(massFractions,view);
    const Real result=pressure/(gasConstant*temperature);
    if(!gasFinite(result) || !(result>Real(0))) return ThermoStatus::NonFiniteInput;
    density=result;
    return ThermoStatus::Success;
}

// Generic caloric inversion permits independent condensed/film calorics to
// share the same safeguarded solve without representing condensed matter as gas.
// Evaluators must validate their full operating interval before this call.
// Both evaluator output and the final temperature are transactional on failure.
template<class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus evaluateCaloricEnergy
(const Evaluator& evaluator,Real temperature,Real& energy,Real& capacity) noexcept
{
    Real localEnergy,localCapacity;
    const ThermoStatus status=evaluator(temperature,localEnergy,localCapacity);
    if(status!=ThermoStatus::Success) return status;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) return ThermoStatus::NonFiniteInput;
    if(!(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;capacity=localCapacity;return ThermoStatus::Success;
}

template<class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus invertCaloricEnergy
(
    const Evaluator& evaluator,Real internalEnergy,Real lower,Real upper,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!gasFinite(internalEnergy)) return ThermoStatus::NonFiniteInput;
    if(controls.maximumIterations<=0 || !gasFinite(controls.minTemperature)
        || !gasFinite(controls.maxTemperature) || controls.minTemperature<Real(0)
        || controls.maxTemperature<Real(0)
        || !gasFinite(controls.absoluteEnergyTolerance) || controls.absoluteEnergyTolerance<Real(0)
        || !gasFinite(controls.relativeEnergyTolerance) || controls.relativeEnergyTolerance<Real(0)
        || !gasFinite(controls.absoluteTemperatureTolerance) || controls.absoluteTemperatureTolerance<Real(0)
        || !gasFinite(controls.relativeTemperatureTolerance) || controls.relativeTemperatureTolerance<Real(0))
        return ThermoStatus::InvalidControls;
    if(!gasFinite(lower) || !gasFinite(upper) || !(lower>Real(0)))
        return ThermoStatus::InvalidControls;
    Real lo=maxValue(lower,controls.minTemperature);
    Real hi=controls.maxTemperature>Real(0)?minValue(upper,controls.maxTemperature):upper;
    if(!(hi>lo)) return ThermoStatus::InvalidControls;
    Real elo,clo,ehi,chi;
    ThermoStatus status=evaluateCaloricEnergy(evaluator,lo,elo,clo);
    if(status!=ThermoStatus::Success) return status;
    status=evaluateCaloricEnergy(evaluator,hi,ehi,chi);
    if(status!=ThermoStatus::Success) return status;
    if(!(ehi>elo)) return ThermoStatus::NonPositiveHeatCapacity;
    // No energy clamp, including for formation-inclusive negative energies.
    if(internalEnergy<elo || internalEnergy>ehi) return ThermoStatus::EnergyOutOfRange;
    if(internalEnergy==elo) { temperature=lo; return ThermoStatus::Success; }
    if(internalEnergy==ehi) { temperature=hi; return ThermoStatus::Success; }
    Real trial=gasFinite(temperature) && temperature>lo && temperature<hi
        ? temperature : lo+Real(0.5)*(hi-lo);
    const Real energyTolerance=controls.absoluteEnergyTolerance
        +controls.relativeEnergyTolerance*absValue(internalEnergy);
    for(int iteration=0;iteration<controls.maximumIterations;++iteration)
    {
        Real energy,capacity;
        status=evaluateCaloricEnergy(evaluator,trial,energy,capacity);
        if(status!=ThermoStatus::Success) return status;
        const Real residual=energy-internalEnergy;
        const Real correction=residual/capacity;
        const Real temperatureTolerance=controls.absoluteTemperatureTolerance
            +controls.relativeTemperatureTolerance*absValue(trial);
        // Both tests are needed: an absolute J tolerance alone accepts a wildly
        // wrong T in a tiny cell. cv converts the residual to an independent K test.
        if(absValue(residual)<=energyTolerance && absValue(correction)<=temperatureTolerance)
        {
            temperature=trial;
            return ThermoStatus::Success;
        }
        if(residual<Real(0)) lo=trial; else hi=trial;
        Real next=trial-correction;
        if(!gasFinite(next) || !(next>lo && next<hi)) next=lo+Real(0.5)*(hi-lo);
        if(next==trial) return ThermoStatus::IterationLimit;
        trial=next;
    }
    return ThermoStatus::IterationLimit;
}

template<class Real,int Ns>
struct MixtureEnergyEvaluator
{
    const Real* speciesMass;
    SpeciesThermoView<Real,Ns> thermo;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    { return evaluateMixtureEnergy(speciesMass,T,thermo,energy,capacity); }
};

template<class Real,int Ns>
UGKWP_GAS_HD inline ThermoStatus invertMixtureEnergy
(
    const Real* speciesMass,Real internalEnergy,const SpeciesThermoView<Real,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    // The generic callback is no substitute for validating every configured
    // species, including zero-inventory products reachable through chemistry.
    const ThermoStatus modelStatus=validateSpeciesThermoView(view);
    if(modelStatus!=ThermoStatus::Success) return modelStatus;
    Real lo=Real(0),hi=(std::numeric_limits<Real>::max)();
    for(int s=0;s<Ns;++s)
    {
        lo=maxValue(lo,view.species[s].minTemperature);
        hi=minValue(hi,view.species[s].maxTemperature);
    }
    const MixtureEnergyEvaluator<Real,Ns> evaluator{speciesMass,view};
    return invertCaloricEnergy(evaluator,internalEnergy,lo,hi,controls,temperature);
}

} // namespace ugkwp
#endif
