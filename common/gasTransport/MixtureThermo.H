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

// Recovery already admits independently transported densities within its
// configured closure tolerance. Construct a disposable composition on the
// simplex from that same accepted inventory, without modifying any inventory.
// Close on the largest component so roundoff cannot create an absent species.
// Invalid inventories are left to the existing caller validation unchanged.
template<class Real,int Ns>
UGKWP_GAS_HD inline bool mixtureMassFractionsFromDensities(
    const Real* partial,Real density,Real closureTolerance,Real* fractions) noexcept
{
    if(!partial||!fractions||!gasFinite(density)||!(density>Real(0))
        ||!gasFinite(closureTolerance)||closureTolerance<Real(0))return false;
    Real total=0;int dependent=0;
    for(int k=0;k<Ns;++k){
        if(!gasFinite(partial[k])||partial[k]<Real(0))return false;
        total+=partial[k];if(partial[k]>partial[dependent])dependent=k;
    }
    if(!gasFinite(total)||!(total>Real(0))
        ||::fabs(total-density)>closureTolerance*density)return false;
    Real local[Ns],sum=0;
    for(int k=0;k<Ns;++k)if(k!=dependent){local[k]=partial[k]/total;sum+=local[k];}
    local[dependent]=Real(1)-sum;
    if(!gasFinite(local[dependent])||local[dependent]<Real(0)||local[dependent]>Real(1))return false;
    for(int k=0;k<Ns;++k)fractions[k]=local[k];
    return true;
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

template<class Real, int Ns, class TableReal, bool Complete=true, bool CompleteEnthalpy=false, bool CompleteEntropy=false>
UGKWP_GAS_HD inline ThermoStatus evaluateSpecies
(
    int species, Real temperature, const SpeciesThermoView<TableReal,Ns>& view,
    SpeciesValues<Real>& output,bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    const bool declared=validMetadata(species,view);
    if (!declared && (!Complete || !view.species || !view.coefficients))
        return ThermoStatus::InvalidModel;
    if (!Complete && !gasFinite(temperature)) return ThermoStatus::NonFiniteInput;
    const SpeciesThermoData<TableReal>& data = view.species[species];
    if(!declared && data.model!=SpeciesThermoModel::LinearCp
        && data.model!=SpeciesThermoModel::NASA7)return ThermoStatus::InvalidModel;
    const TableReal* c = view.coefficients + data.coefficientOffset;
    const Real R = universalGasConstant<Real>()/data.molarMass;
    const int available=declared?view.coefficientCount-data.coefficientOffset:0;


    bool completeCoefficients=!declared;
    ThermoStatus failure=!declared?ThermoStatus::InvalidModel
        : (Complete && !gasFinite(temperature)?ThermoStatus::NonFiniteInput:ThermoStatus::Success);
    SpeciesValues<Real> local;
    if (data.model == SpeciesThermoModel::LinearCp)
    {
        local.cp = c[0] + c[1]*temperature;



        if(Complete && CompleteEnthalpy && !completeCoefficients && available<3
            && failure==ThermoStatus::Success)failure=ThermoStatus::NonFiniteInput;
        local.h = completeCoefficients || (Complete && CompleteEnthalpy) || available>=3
            ? c[2] + temperature*(c[0] + Real(0.5)*c[1]*temperature) : invalidValue<Real>();
        local.entropy = data.hasEntropyReference || (Complete && CompleteEntropy)
            ? data.referenceEntropy + c[0]*::log(temperature/data.referenceTemperature)
                + c[1]*(temperature-data.referenceTemperature)
            : invalidValue<Real>();
    }
    else
    {
        const int segment=temperature>data.midTemperature?7:0;
        if(!completeCoefficients && available-segment<5) {
            if(!Complete)return ThermoStatus::InvalidModel;
            completeCoefficients=true;
            if(failure==ThermoStatus::Success)failure=ThermoStatus::InvalidModel;
        }
        c+=segment;
        local.cp = R*(c[0]+temperature*(c[1]+temperature*(c[2]+temperature*(c[3]+temperature*c[4]))));
        if(Complete && CompleteEnthalpy && !completeCoefficients && available-segment<6
            && failure==ThermoStatus::Success)failure=ThermoStatus::NonFiniteInput;
        local.h = completeCoefficients || (Complete && CompleteEnthalpy) || available-segment>=6
            ? R*(c[5]+temperature*(c[0]+temperature*(c[1]/Real(2)
                +temperature*(c[2]/Real(3)+temperature*(c[3]/Real(4)+temperature*c[4]/Real(5))))))
            : invalidValue<Real>();
        local.entropy = completeCoefficients || (Complete && CompleteEntropy) || available-segment>=7
            ? R*(c[0]*::log(temperature)+c[6]+temperature*(c[1]
                +temperature*(c[2]/Real(2)+temperature*(c[3]/Real(3)+temperature*c[4]/Real(4)))))
            : invalidValue<Real>();
    }
    local.cv = local.cp-R;
    local.e = local.h-R*temperature;


    output = local;
    if(evaluated)*evaluated=true;
    return failure;
}

} // namespace gasThermoDetail


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
    return gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false>(species,temperature,view,values)==ThermoStatus::Success \
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
    if(gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false>(species,temperature,view,values)!=ThermoStatus::Success)
        return gasThermoDetail::invalidValue<Real>();
    const Real gibbs=values.h-temperature*values.entropy;
    return gasFinite(gibbs)?gibbs:gasThermoDetail::invalidValue<Real>();
}

// The arrays contain masses [kg], not mass fractions or molar concentrations.
// Supplying mass fractions is valid only when intentionally evaluating one kg.
template<bool Refinement=false,class Real,int Ns,class TableReal,bool Complete=!Refinement,bool CompleteEnthalpy=false>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureEnergy
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view,
    Real& energy,Real& heatCapacity,bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    if(!speciesMass) return ThermoStatus::InvalidComposition;
    Real totalMass=Real(0),localEnergy=Real(0),localCapacity=Real(0);
    ThermoStatus failure=ThermoStatus::Success;
    for(int s=0;s<Ns;++s)
    {
        if(!gasFinite(speciesMass[s]) || (Refinement && speciesMass[s]<Real(0))) {
            if(!Complete || Refinement) return ThermoStatus::InvalidComposition;
            if(failure==ThermoStatus::Success)failure=ThermoStatus::InvalidComposition;
        }
        gasThermoDetail::SpeciesValues<Real> values;
        bool speciesEvaluated=false;
        const ThermoStatus status=gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,Complete && !Refinement,CompleteEnthalpy && !Refinement>
            (s,temperature,view,values,Complete && !Refinement?&speciesEvaluated:nullptr);
        if(status!=ThermoStatus::Success) {
            if(!Complete || Refinement || !speciesEvaluated) return status;
            if(failure==ThermoStatus::Success)failure=status;
        }
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
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) {
        if(!Complete || Refinement)return ThermoStatus::NonFiniteInput;
        if(failure==ThermoStatus::Success)failure=ThermoStatus::NonFiniteInput;
    }
    if(Refinement && !(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;
    heatCapacity=localCapacity;
    if(evaluated)*evaluated=true;
    return failure;
}



template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureInternalEnergy
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view,
    Real& energy,bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    if(!speciesMass)return ThermoStatus::InvalidComposition;
    Real localEnergy=Real(0);ThermoStatus failure=ThermoStatus::Success;
    for(int species=0;species<Ns;++species)
    {
        if(!gasFinite(speciesMass[species]) && failure==ThermoStatus::Success)
            failure=ThermoStatus::InvalidComposition;
        gasThermoDetail::SpeciesValues<Real> values;bool propertyEvaluated=false;
        const auto status=gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,true,true>
            (species,temperature,view,values,&propertyEvaluated);
        if(!propertyEvaluated)return status;
        if(status!=ThermoStatus::Success && failure==ThermoStatus::Success)failure=status;
        localEnergy+=speciesMass[species]*values.e;
    }
    if(!gasFinite(localEnergy) && failure==ThermoStatus::Success)failure=ThermoStatus::NonFiniteInput;
    energy=localEnergy;if(evaluated)*evaluated=true;return failure;
}



template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureHeatCapacity
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view,
    Real& heatCapacity,bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    if(!speciesMass)return ThermoStatus::InvalidComposition;
    Real localCapacity=Real(0);ThermoStatus failure=ThermoStatus::Success;
    for(int species=0;species<Ns;++species)
    {
        if(!gasFinite(speciesMass[species]) && failure==ThermoStatus::Success)
            failure=ThermoStatus::InvalidComposition;
        gasThermoDetail::SpeciesValues<Real> values;bool propertyEvaluated=false;
        const auto status=gasThermoDetail::evaluateSpecies
            (species,temperature,view,values,&propertyEvaluated);
        if(!propertyEvaluated)return status;
        if(status!=ThermoStatus::Success && failure==ThermoStatus::Success)failure=status;
        localCapacity+=speciesMass[species]*values.cv;
    }
    if(!gasFinite(localCapacity) && failure==ThermoStatus::Success)failure=ThermoStatus::NonFiniteInput;
    heatCapacity=localCapacity;if(evaluated)*evaluated=true;return failure;
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

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real mixtureGasConstant
(const Real* speciesMass,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    if(!speciesMass || !view.species)
        return gasThermoDetail::invalidValue<Real>();
    Real totalMass=Real(0),weightedConstant=Real(0);
    for(int species=0;species<Ns;++species)
    {
        totalMass+=speciesMass[species];
        weightedConstant+=speciesMass[species]*universalGasConstant<Real>()/view.species[species].molarMass;
    }
    const Real result=weightedConstant/totalMass;
    return result;
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
    if(!view.species) return ThermoStatus::InvalidModel;
    const Real gasConstant=mixtureGasConstant(massFractions,view);
    const Real result=pressure/(gasConstant*temperature);
    if(!gasFinite(result) || (Refinement && !(result>Real(0)))) {
        if(!Refinement)density=result;
        return ThermoStatus::NonFiniteInput;
    }
    density=result;
    return ThermoStatus::Success;
}


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


template<bool Refinement=false,class Real,class Evaluator,bool Complete=!Refinement>
UGKWP_GAS_HD inline ThermoStatus evaluateCaloricEnergy
(const Evaluator& evaluator,Real temperature,Real& energy,Real& capacity,
 bool* evaluated=nullptr) noexcept
{
    Real localEnergy=gasThermoDetail::invalidValue<Real>();
    Real localCapacity=gasThermoDetail::invalidValue<Real>();
    const ThermoStatus status=evaluator(temperature,localEnergy,localCapacity);
    if(evaluated)*evaluated=status==ThermoStatus::Success;
    if(status!=ThermoStatus::Success) return status;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) {
        if(!Refinement && (Complete || evaluated)){energy=localEnergy;capacity=localCapacity;}
        return ThermoStatus::NonFiniteInput;
    }
    if(Refinement && !(localCapacity>Real(0))) return ThermoStatus::NonPositiveHeatCapacity;
    energy=localEnergy;capacity=localCapacity;return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal,bool Refinement> struct MixtureEnergyEvaluator;



template<class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus completeCaloricEvaluation
(const Evaluator& evaluator,Real T,Real& energy,Real& capacity,bool& evaluated,
 ThermoStatus previous=ThermoStatus::Success) noexcept
{
    if(previous!=ThermoStatus::Success){evaluated=false;return previous;}
    const ThermoStatus status=evaluator(T,energy,capacity);
    evaluated=status==ThermoStatus::Success;
    return status;
}
template<class Real,int Ns,class TableReal,bool Refinement>
UGKWP_GAS_HD inline ThermoStatus completeCaloricEvaluation
(const MixtureEnergyEvaluator<Real,Ns,TableReal,Refinement>& evaluator,
 Real T,Real& energy,Real& capacity,bool& evaluated,
 ThermoStatus =ThermoStatus::Success) noexcept
{ return evaluator.complete(T,energy,capacity,evaluated); }



template<class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus completeNonFiniteCaloricTarget
(const Evaluator&,Real,Real,const ThermoInversionControls<Real>&,Real&,bool*) noexcept
{ return ThermoStatus::NonFiniteInput; }
template<class Real,int Ns,class TableReal,bool Refinement>
UGKWP_GAS_HD inline ThermoStatus completeNonFiniteCaloricTarget
(
    const MixtureEnergyEvaluator<Real,Ns,TableReal,Refinement>& evaluator,
    Real lower,Real upper,const ThermoInversionControls<Real>& controls,
    Real& temperature,bool* evaluatedTemperature
) noexcept
{
    using namespace gasThermoDetail;
    const Real lo=maxValue(lower,controls.minTemperature);
    const Real hi=controls.maxTemperature>Real(0)?minValue(upper,controls.maxTemperature):upper;
    const Real trial=gasFinite(temperature)?temperature:lo+Real(0.5)*(hi-lo);
    Real energy,capacity;bool evaluated=false;
    const ThermoStatus status=evaluator.complete(trial,energy,capacity,evaluated);
    if(!evaluated)return status;
    temperature=trial;
    if(evaluatedTemperature)*evaluatedTemperature=true;
    return ThermoStatus::NonFiniteInput;
}

template<bool Refinement=false,class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus invertCaloricEnergy
(
    const Evaluator& evaluator,Real internalEnergy,Real lower,Real upper,
    const ThermoInversionControls<Real>& controls,Real& temperature,bool* evaluatedTemperature=nullptr
) noexcept
{
    using namespace gasThermoDetail;
    if(evaluatedTemperature)*evaluatedTemperature=false;
    if(!gasFinite(internalEnergy)) {
        if(!Refinement)return completeNonFiniteCaloricTarget
            (evaluator,lower,upper,controls,temperature,evaluatedTemperature);
        return ThermoStatus::NonFiniteInput;
    }
    Real lo=maxValue(lower,controls.minTemperature);
    Real hi=controls.maxTemperature>Real(0)?minValue(upper,controls.maxTemperature):upper;
    Real elo=invalidValue<Real>(),clo=invalidValue<Real>();
    Real ehi=invalidValue<Real>(),chi=invalidValue<Real>();
    const ThermoStatus lowStatus=evaluateCaloricEnergy<Refinement,Real,Evaluator,false>(evaluator,lo,elo,clo);
    const ThermoStatus highStatus=evaluateCaloricEnergy<Refinement,Real,Evaluator,false>(evaluator,hi,ehi,chi);
    const bool lowAvailable=lowStatus==ThermoStatus::Success;
    const bool highAvailable=highStatus==ThermoStatus::Success;


    if(Refinement) {
        if(!lowAvailable)return lowStatus;
        if(!highAvailable)return highStatus;
        if(!(ehi>elo))return ThermoStatus::NonPositiveHeatCapacity;
        if(internalEnergy<elo || internalEnergy>ehi)return ThermoStatus::EnergyOutOfRange;
    }
    if(lowAvailable && internalEnergy==elo) { temperature=lo;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success; }
    if(highAvailable && internalEnergy==ehi) { temperature=hi;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success; }
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
        Real energy,capacity;bool evaluated=false;
        const ThermoStatus status=evaluateCaloricEnergy<Refinement,Real,Evaluator,false>
            (evaluator,trial,energy,capacity,Refinement?nullptr:&evaluated);
        if(status!=ThermoStatus::Success)
        {
            if(!Refinement && available) { temperature=saved;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success; }
            if(!Refinement) {
                if(!evaluated) {
                    const ThermoStatus completed=completeCaloricEvaluation(evaluator,trial,energy,capacity,evaluated,status);
                    if(!evaluated)return completed;
                }
                temperature=trial;
                if(evaluatedTemperature)*evaluatedTemperature=true;
            }
            return status;
        }
        saved=trial;available=true;
        const Real residual=energy-internalEnergy;
        const Real correction=residual/capacity;
        const Real temperatureTolerance=controls.absoluteTemperatureTolerance
            +controls.relativeTemperatureTolerance*absValue(trial);
        if((!Refinement && residual==Real(0)) || (absValue(residual)<=energyTolerance
            && absValue(correction)<=temperatureTolerance)) { temperature=trial;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success; }
        if(bracket)
        {

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

    temperature=saved;
    if(evaluatedTemperature)*evaluatedTemperature=true;
    return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal=Real,bool Refinement=false>
struct MixtureEnergyEvaluator
{
    const Real* speciesMass;
    SpeciesThermoView<TableReal,Ns> thermo;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    { return evaluateMixtureEnergy<Refinement,Real,Ns,TableReal,false>(speciesMass,T,thermo,energy,capacity); }
    UGKWP_GAS_HD ThermoStatus complete(Real T,Real& energy,Real& capacity,bool& evaluated) const noexcept
    {
        return evaluateMixtureEnergy<false,Real,Ns,TableReal,true>(speciesMass,T,thermo,energy,capacity,&evaluated);
    }
};

template<bool Refinement=false,class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus invertMixtureEnergy
(
    const Real* speciesMass,Real internalEnergy,const SpeciesThermoView<TableReal,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature,bool* evaluatedTemperature=nullptr
) noexcept
{
    using namespace gasThermoDetail;
    if(evaluatedTemperature)*evaluatedTemperature=false;
    if(!view.species || !speciesMass) return ThermoStatus::InvalidModel;
    Real lo=Real(0),hi=(std::numeric_limits<Real>::max)();
    for(int s=0;s<Ns;++s)
    {
        lo=maxValue(lo,Real(view.species[s].minTemperature));
        hi=minValue(hi,Real(view.species[s].maxTemperature));
    }
    const MixtureEnergyEvaluator<Real,Ns,TableReal,Refinement> evaluator{speciesMass,view};
    return invertCaloricEnergy<Refinement>(evaluator,internalEnergy,lo,hi,controls,temperature,evaluatedTemperature);
}

} // namespace ugkwp
#endif
