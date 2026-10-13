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
    // This is an addressability check. Temperature spans and physical-property
    // proofs do not decide whether an existing formula can be evaluated.
    const int count = data.model == SpeciesThermoModel::LinearCp ? 2
        : (data.model == SpeciesThermoModel::NASA7 ? 5 : 0);
    if (!count || view.coefficientCount < count || data.coefficientOffset < 0
        || data.coefficientOffset > view.coefficientCount-count) return false;
    return true;
}

template<class Real>
struct SpeciesValues
{
    Real cp, cv, h, e, entropy;
};

template<class Real, int Ns, class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateSpecies
(
    int species, Real temperature, const SpeciesThermoView<TableReal,Ns>& view,
    SpeciesValues<Real>& output
) noexcept
{
    if (!validMetadata(species,view)) return ThermoStatus::InvalidModel;
    if (!gasFinite(temperature)) return ThermoStatus::NonFiniteInput;
    const SpeciesThermoData<TableReal>& data = view.species[species];
    const TableReal* c = view.coefficients + data.coefficientOffset;
    const Real R = universalGasConstant<Real>()/data.molarMass;
    const int available=view.coefficientCount-data.coefficientOffset;
    SpeciesValues<Real> local;
    if (data.model == SpeciesThermoModel::LinearCp)
    {
        local.cp = c[0] + c[1]*temperature;
        local.h = available>=3
            ? c[2] + temperature*(c[0] + Real(0.5)*c[1]*temperature) : invalidValue<Real>();
        local.entropy = data.hasEntropyReference
            ? data.referenceEntropy + c[0]*::log(temperature/data.referenceTemperature)
                + c[1]*(temperature-data.referenceTemperature)
            : invalidValue<Real>();
    }
    else
    {
        const int segment=temperature>data.midTemperature?7:0;
        if(available-segment<5)return ThermoStatus::InvalidModel;
        c+=segment;
        local.cp = R*(c[0]+temperature*(c[1]+temperature*(c[2]+temperature*(c[3]+temperature*c[4]))));
        local.h = available-segment>=6
            ? R*(c[5]+temperature*(c[0]+temperature*(c[1]/Real(2)
                +temperature*(c[2]/Real(3)+temperature*(c[3]/Real(4)+temperature*c[4]/Real(5))))))
            : invalidValue<Real>();
        local.entropy = available-segment>=7
            ? R*(c[0]*::log(temperature)+c[6]+temperature*(c[1]
                +temperature*(c[2]/Real(2)+temperature*(c[3]/Real(3)+temperature*c[4]/Real(4)))))
            : invalidValue<Real>();
    }
    local.cv = local.cp-R;
    local.e = local.h-R*temperature;
    // Availability is per requested property. An unused entropy or overflowing
    // enthalpy must not discard a finite cp. Consumers check their own channel.
    output = local;
    return ThermoStatus::Success;
}

} // namespace gasThermoDetail

// Inspect readable species declarations, without whole-range property proofs.
template<class Real, int Ns>
UGKWP_GAS_HD inline ThermoStatus validateSpeciesThermoView
(const SpeciesThermoView<Real,Ns>& view) noexcept
{
    for (int s=0;s<Ns;++s)
        if (!gasThermoDetail::validMetadata(s,view)) return ThermoStatus::InvalidModel;
    return ThermoStatus::Success;
}

#define UGKWP_DEFINE_SPECIES_PROPERTY(Name, Member) \
template<class Real, int Ns, class TableReal> \
UGKWP_GAS_HD inline Real Name \
(int species,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept \
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

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real speciesGibbsStandard
(int species,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    gasThermoDetail::SpeciesValues<Real> values;
    if(gasThermoDetail::evaluateSpecies(species,temperature,view,values)!=ThermoStatus::Success)
        return gasThermoDetail::invalidValue<Real>();
    const Real gibbs=values.h-temperature*values.entropy;
    return gasFinite(gibbs)?gibbs:gasThermoDetail::invalidValue<Real>();
}

// The arrays contain masses [kg], not mass fractions or molar concentrations.
// Supplying mass fractions is valid only when intentionally evaluating one kg.
template<bool Refinement=false,class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureEnergy
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view,
    Real& energy,Real& heatCapacity
) noexcept
{
    if(!speciesMass) return ThermoStatus::InvalidComposition;
    Real totalMass=Real(0),localEnergy=Real(0),localCapacity=Real(0);
    for(int s=0;s<Ns;++s)
    {
        if(!gasFinite(speciesMass[s]) || (Refinement && speciesMass[s]<Real(0)))
            return ThermoStatus::InvalidComposition;
        gasThermoDetail::SpeciesValues<Real> values;
        const ThermoStatus status=gasThermoDetail::evaluateSpecies(s,temperature,view,values);
        if(status!=ThermoStatus::Success) return status;
        if(Refinement) {
            if(temperature<view.species[s].minTemperature || temperature>view.species[s].maxTemperature)
                return ThermoStatus::TemperatureOutOfRange;
            if(!gasFinite(values.cp) || !gasFinite(values.cv) || !gasFinite(values.h) || !gasFinite(values.e))
                return ThermoStatus::NonFiniteInput;
            if(!(values.cv>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
            totalMass+=speciesMass[s];
        }
        localEnergy+=speciesMass[s]*values.e;
        localCapacity+=speciesMass[s]*values.cv;
    }
    if(Refinement && (!gasFinite(totalMass) || !(totalMass>Real(0)))) return ThermoStatus::InvalidComposition;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) return ThermoStatus::NonFiniteInput;
    if(Refinement && !(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;
    heatCapacity=localCapacity;
    return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real mixtureEnergy
(const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    if(!speciesMass)return gasThermoDetail::invalidValue<Real>();
    Real energy=0;
    for(int s=0;s<Ns;++s)energy+=speciesMass[s]*speciesE(s,temperature,view);
    return gasFinite(energy)?energy:gasThermoDetail::invalidValue<Real>();
}

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real mixtureHeatCapacity
(const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    if(!speciesMass)return gasThermoDetail::invalidValue<Real>();
    Real capacity=0;
    for(int s=0;s<Ns;++s)capacity+=speciesMass[s]*speciesCv(s,temperature,view);
    return gasFinite(capacity)?capacity:gasThermoDetail::invalidValue<Real>();
}

// Mass-specific mixture R. Input may contain masses or mass fractions; total
// mass is explicitly divided out. Invalid/zero mass has no meaningful R.
template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real mixtureGasConstant
(const Real* speciesMass,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    if(!speciesMass || !view.species)
        return gasThermoDetail::invalidValue<Real>();
    Real totalMass=Real(0),weightedConstant=Real(0);
    for(int species=0;species<Ns;++species)
    {
        if(!gasFinite(speciesMass[species]))
            return gasThermoDetail::invalidValue<Real>();
        totalMass+=speciesMass[species];
        weightedConstant+=speciesMass[species]*universalGasConstant<Real>()/view.species[species].molarMass;
    }
    if(!gasFinite(totalMass) || totalMass==Real(0) || !gasFinite(weightedConstant))
        return gasThermoDetail::invalidValue<Real>();
    const Real result=weightedConstant/totalMass;
    return gasFinite(result)?result:gasThermoDetail::invalidValue<Real>();
}

// p/T/Y initialization has one common EOS owner. Y must already be admissible;
// this routine never repairs species or silently accepts arbitrary mass units.
// All species (including zero-mass products) must share the supplied T range.
template<bool Refinement=false,class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus mixtureDensityFromPressureTemperatureImpl
(
    Real pressure,Real temperature,const Real* massFractions,
    const SpeciesThermoView<TableReal,Ns>& view,Real& density,
    Real compositionTolerance=Real(64)*std::numeric_limits<Real>::epsilon()
) noexcept
{
    if(Refinement)
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
        const ThermoStatus status=evaluateMixtureEnergy<true>(massFractions,temperature,view,energy,capacity);
        if(status!=ThermoStatus::Success) return status;
    }
    if(!massFractions) return ThermoStatus::InvalidComposition;
    const Real gasConstant=mixtureGasConstant(massFractions,view);
    const Real result=pressure/(gasConstant*temperature);
    if(!gasFinite(result) || (Refinement && !(result>Real(0)))) return ThermoStatus::NonFiniteInput;
    density=result;
    return ThermoStatus::Success;
}

// Preserve the public explicit <Real,Ns,TableReal> entry-point spelling.
template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus mixtureDensityFromPressureTemperature
(
    Real pressure,Real temperature,const Real* massFractions,
    const SpeciesThermoView<TableReal,Ns>& view,Real& density,
    Real compositionTolerance=Real(64)*std::numeric_limits<Real>::epsilon()
) noexcept
{
    return mixtureDensityFromPressureTemperatureImpl<false>(pressure,temperature,massFractions,view,density,compositionTolerance);
}

// Generic caloric inversion permits independent condensed/film calorics to
// share the same safeguarded solve without representing condensed matter as gas.
// Evaluators must validate their full operating interval before this call.
// Both evaluator output and the final temperature are transactional on failure.
template<bool Refinement=false,class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus evaluateCaloricEnergy
(const Evaluator& evaluator,Real temperature,Real& energy,Real& capacity) noexcept
{
    Real localEnergy=gasThermoDetail::invalidValue<Real>();
    Real localCapacity=gasThermoDetail::invalidValue<Real>();
    const ThermoStatus status=evaluator(temperature,localEnergy,localCapacity);
    if(status!=ThermoStatus::Success) return status;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) return ThermoStatus::NonFiniteInput;
    if(Refinement && !(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;capacity=localCapacity;return ThermoStatus::Success;
}

template<bool Refinement=false,class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus invertCaloricEnergy
(
    const Evaluator& evaluator,Real internalEnergy,Real lower,Real upper,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!gasFinite(internalEnergy)) return ThermoStatus::NonFiniteInput;
    Real lo=maxValue(lower,controls.minTemperature);
    Real hi=controls.maxTemperature>Real(0)?minValue(upper,controls.maxTemperature):upper;
    Real elo=invalidValue<Real>(),clo=invalidValue<Real>();
    Real ehi=invalidValue<Real>(),chi=invalidValue<Real>();
    const ThermoStatus lowStatus=evaluateCaloricEnergy<Refinement>(evaluator,lo,elo,clo);
    const ThermoStatus highStatus=evaluateCaloricEnergy<Refinement>(evaluator,hi,ehi,chi);
    const bool lowAvailable=lowStatus==ThermoStatus::Success;
    const bool highAvailable=highStatus==ThermoStatus::Success;
    // These are original trial failures: the reactor must damp/refine them,
    // rather than publish a different finite temperature into a successful step.
    if(Refinement) {
        if(!lowAvailable)return lowStatus;
        if(!highAvailable)return highStatus;
        if(!(ehi>elo))return ThermoStatus::NonPositiveHeatCapacity;
        if(internalEnergy<elo || internalEnergy>ehi)return ThermoStatus::EnergyOutOfRange;
    }
    if(lowAvailable && internalEnergy==elo) { temperature=lo; return ThermoStatus::Success; }
    if(highAvailable && internalEnergy==ehi) { temperature=hi; return ThermoStatus::Success; }
    bool bracket=lowAvailable && highAvailable && lo<hi
        && ((elo<internalEnergy && ehi>internalEnergy)
            || (elo>internalEnergy && ehi<internalEnergy));
    Real trial=gasFinite(temperature)
        && ((Refinement ? false : (!bracket || controls.maximumIterations<=0)) || (temperature>lo && temperature<hi))
        ? temperature : lo+Real(0.5)*(hi-lo);
    Real saved=lowAvailable?lo:hi;
    bool available=lowAvailable || highAvailable;
    const Real energyTolerance=controls.absoluteEnergyTolerance
        +controls.relativeEnergyTolerance*absValue(internalEnergy);
    const int iterations=controls.maximumIterations>0?controls.maximumIterations:1;
    for(int iteration=0;iteration<iterations;++iteration)
    {
        Real energy,capacity;
        const ThermoStatus status=evaluateCaloricEnergy<Refinement>(evaluator,trial,energy,capacity);
        if(status!=ThermoStatus::Success)
        {
            if(!Refinement && available) { temperature=saved; return ThermoStatus::Success; }
            return status;
        }
        saved=trial;available=true;
        const Real residual=energy-internalEnergy;
        const Real correction=residual/capacity;
        const Real temperatureTolerance=controls.absoluteTemperatureTolerance
            +controls.relativeTemperatureTolerance*absValue(trial);
        if((!Refinement && residual==Real(0)) || (absValue(residual)<=energyTolerance
            && absValue(correction)<=temperatureTolerance)) { temperature=trial;return ThermoStatus::Success; }
        if(bracket)
        {
            // Endpoint residual signs also support decreasing calorics.
            if((residual<Real(0))==(elo<internalEnergy)) { lo=trial;elo=energy; }
            else { hi=trial;ehi=energy; }
        }
        Real next=trial-correction;
        if(bracket && (!gasFinite(next) || !(next>lo && next<hi)))
            next=lo+Real(0.5)*(hi-lo);
        if(!gasFinite(next) || next==trial) { if(Refinement)return ThermoStatus::IterationLimit; break; }
        trial=next;
    }
    if(Refinement)return ThermoStatus::IterationLimit;
    // saved is an evaluated T/E pair, never the next unevaluated proposal.
    temperature=saved;
    return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal=Real,bool Refinement=false>
struct MixtureEnergyEvaluator
{
    const Real* speciesMass;
    SpeciesThermoView<TableReal,Ns> thermo;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    { return evaluateMixtureEnergy<Refinement>(speciesMass,T,thermo,energy,capacity); }
};

template<bool Refinement=false,class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus invertMixtureEnergy
(
    const Real* speciesMass,Real internalEnergy,const SpeciesThermoView<TableReal,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!view.species || !speciesMass) return ThermoStatus::InvalidModel;
    Real lo=Real(0),hi=(std::numeric_limits<Real>::max)();
    for(int s=0;s<Ns;++s)
    {
        lo=maxValue(lo,Real(view.species[s].minTemperature));
        hi=minValue(hi,Real(view.species[s].maxTemperature));
    }
    const MixtureEnergyEvaluator<Real,Ns,TableReal,Refinement> evaluator{speciesMass,view};
    return invertCaloricEnergy<Refinement>(evaluator,internalEnergy,lo,hi,controls,temperature);
}

} // namespace ugkwp
#endif
