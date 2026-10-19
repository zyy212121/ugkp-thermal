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

// Fractions preserve independently transported species inventories. A closure
// residual is diagnostic only; finite component clipping never renormalizes Y.
template<class Real,int Ns>
UGKWP_GAS_HD inline bool mixtureMassFractionsFromDensities(
    const Real* partial,Real density,Real closureTolerance,Real* fractions,
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr) noexcept
{
    using namespace ugkp::diagnostics;
    if(!partial || !fractions) return false;
    if(!gasFinite(density) || density==Real(0)) {
        if(diagnostics)recordFatal(diagnostics,makeRecord(!gasFinite(density)?Code::NonFinite:Code::ZeroDenominator,2301,0,0,EntityKind::None,-1,-1,density,0,0,Quantity::Density));
        return false;
    }
    Real total=0;
    for(int k=0;k<Ns;++k){
        const Real value=partial[k]/density;
        if(!gasFinite(value)) {
            if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,10000,0,0,EntityKind::Species,k,k,value,0,0,Quantity::MassFraction));
            return false;
        }
        total+=value;
        if(!gasFinite(total)) {
            recordWarning(diagnostics,makeRecord(Code::ConservationResidual,10001,0,0,EntityKind::Species,k,k,total,1,total,Quantity::Conservation));
            // Preserve bounded fractions; total is only a closure diagnostic.
        }
        const Real clipped=value<Real(0)?Real(0):(value>Real(1)?Real(1):value);
        if(clipped!=value)recordWarning(diagnostics,makeRecord(Code::FiniteYClip,10002,0,0,EntityKind::Species,k,k,value,0,clipped,Quantity::MassFraction));
        fractions[k]=clipped;
    }
    if(!gasFinite(total) || ::fabs(total-Real(1))>closureTolerance)
        recordWarning(diagnostics,makeRecord(Code::ConservationResidual,10003,0,0,EntityKind::None,-1,-1,total,1,total,Quantity::Conservation));
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

// POD failures propagate to the owning host/device operation. Optional context
// records the same failure immediately; no alternate property evaluation occurs.
UGKWP_GAS_HD inline ThermoStatus failure(ugkp::diagnostics::DeviceContext* ctx,
    ThermoStatus status,ugkp::diagnostics::Code code,int species,double value,
    unsigned quantity=ugkp::diagnostics::Quantity::Unspecified) noexcept
{
    if(ctx)ugkp::diagnostics::recordFatal(ctx,ugkp::diagnostics::makeRecord(code,2302,0,0,
        ugkp::diagnostics::EntityKind::Species,species,species,value,0,0,quantity));
    return status;
}

template<class Real, int Ns, class TableReal, bool Complete=true, bool CompleteEnthalpy=false, bool CompleteEntropy=false>
UGKWP_GAS_HD inline ThermoStatus evaluateSpecies
(
    int species, Real temperature, const SpeciesThermoView<TableReal,Ns>& view,
    SpeciesValues<Real>& output,bool* evaluated=nullptr,Real* nonfiniteValue=nullptr
) noexcept
{
    using namespace ugkp::diagnostics;
    if(evaluated)*evaluated=false;
    if(species<0 || species>=Ns)
        return failure(view.diagnostics,ThermoStatus::InvalidModel,Code::InvalidIndex,species,species,Quantity::Index);
    if(!validMetadata(species,view))
        return failure(view.diagnostics,ThermoStatus::InvalidModel,Code::MissingOperator,species,0);
    if(!gasFinite(temperature))
        return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,species,temperature,Quantity::Temperature);
    if(temperature<Real(0))
        return failure(view.diagnostics,ThermoStatus::TemperatureOutOfRange,Code::NegativeTemperature,species,temperature,Quantity::Temperature);
    const auto& data=view.species[species];
    if(CompleteEntropy && data.model==SpeciesThermoModel::LinearCp && !data.hasEntropyReference)
        return failure(view.diagnostics,ThermoStatus::InvalidModel,Code::MissingOperator,species,0);
    if(!gasFinite(data.molarMass) || data.molarMass==Real(0))
        return failure(view.diagnostics,ThermoStatus::InvalidModel,!gasFinite(data.molarMass)?Code::NonFinite:Code::ZeroDenominator,species,data.molarMass,Quantity::Denominator);
    const Real R=universalGasConstant<Real>()/data.molarMass;
    const int segment=data.model==SpeciesThermoModel::NASA7 && temperature>data.midTemperature?7:0;
    const int available=view.coefficientCount-data.coefficientOffset-segment;
    const int required=data.model==SpeciesThermoModel::LinearCp?2:5;
    if(available<required || (CompleteEnthalpy && available<(required+1)) || (CompleteEntropy && data.model==SpeciesThermoModel::NASA7 && available<7))
        return failure(view.diagnostics,ThermoStatus::InvalidModel,Code::InvalidIndex,species,available,Quantity::Index);
    const TableReal* c=view.coefficients+data.coefficientOffset+segment;
    SpeciesValues<Real> local;
    if(data.model==SpeciesThermoModel::LinearCp) {
        local.cp=c[0]+c[1]*temperature;
        local.h=available>=3?c[2]+temperature*(c[0]+Real(.5)*c[1]*temperature):invalidValue<Real>();
        local.entropy=invalidValue<Real>();
        if(data.hasEntropyReference) {
            if(temperature==Real(0) || data.referenceTemperature<=Real(0)) {
                if(CompleteEntropy)
                    return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::ZeroDenominator,species,temperature,Quantity::Temperature);
            } else {
                local.entropy=data.referenceEntropy+c[0]*::log(temperature/data.referenceTemperature)+c[1]*(temperature-data.referenceTemperature);
            }
        }
    } else {
        local.cp=R*(c[0]+temperature*(c[1]+temperature*(c[2]+temperature*(c[3]+temperature*c[4]))));
        local.h=available>=6?R*(c[5]+temperature*(c[0]+temperature*(c[1]/Real(2)+temperature*(c[2]/Real(3)+temperature*(c[3]/Real(4)+temperature*c[4]/Real(5)))))):invalidValue<Real>();
        local.entropy=available>=7 && temperature>Real(0)?R*(c[0]*::log(temperature)+c[6]+temperature*(c[1]+temperature*(c[2]/Real(2)+temperature*(c[3]/Real(3)+temperature*c[4]/Real(4))))):invalidValue<Real>();
    }
    local.cv=local.cp-R;local.e=local.h-R*temperature;
    if(!gasFinite(local.cp) || !gasFinite(local.cv) || (CompleteEnthalpy && !gasFinite(local.h)) || (CompleteEntropy && !gasFinite(local.entropy))) {
        const Real observed=!gasFinite(local.cp)?local.cp:(!gasFinite(local.cv)?local.cv:((CompleteEnthalpy && !gasFinite(local.h))?local.h:local.entropy));
        // A face producer can retain the actual property fault while supplying
        // its own entity/operator context; failed output remains unpublished.
        if(nonfiniteValue)*nonfiniteValue=observed;
        return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,species,observed);
    }
    output=local;if(evaluated)*evaluated=true;
    return ThermoStatus::Success;
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

#define UGKWP_DEFINE_SPECIES_PROPERTY(Name, Member, NeedH, NeedS) \
template<class Real, int Ns, class TableReal> \
UGKWP_GAS_HD inline Real Name \
(int species,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept \
{ \
    gasThermoDetail::SpeciesValues<Real> values{}; \
    if(gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,NeedH,NeedS>(species,temperature,view,values)!=ThermoStatus::Success) \
        return gasThermoDetail::invalidValue<Real>(); \
    if(!gasFinite(values.Member)) { \
        gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,ugkp::diagnostics::Code::NonFinite,species,values.Member); \
        return gasThermoDetail::invalidValue<Real>(); \
    } \
    return values.Member; \
}
UGKWP_DEFINE_SPECIES_PROPERTY(speciesCp,cp,false,false)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesCv,cv,false,false)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesH,h,true,false)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesE,e,true,false)
UGKWP_DEFINE_SPECIES_PROPERTY(speciesEntropyStandard,entropy,false,true)
#undef UGKWP_DEFINE_SPECIES_PROPERTY

// Joint channels use exactly the same evaluator and arithmetic as speciesH/Cp.
// A failed H channel still evaluates Cp as before, retaining the first fatal
// record and the Cp value consumed by the caller's contextual diagnostic.
template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline void speciesCpH(int species,Real temperature,
 const SpeciesThermoView<TableReal,Ns>& view,Real& cp,Real& h) noexcept {
    gasThermoDetail::SpeciesValues<Real> values{};
    if(gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,true,false>(species,temperature,view,values)!=ThermoStatus::Success){
        h=gasThermoDetail::invalidValue<Real>();cp=speciesCp(species,temperature,view);return;
    }
    h=values.h;cp=values.cp;
}

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline Real speciesGibbsStandard
(int species,Real temperature,const SpeciesThermoView<TableReal,Ns>& view) noexcept
{
    gasThermoDetail::SpeciesValues<Real> values;
    if(gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,true,true>(species,temperature,view,values)!=ThermoStatus::Success)
        return gasThermoDetail::invalidValue<Real>();
    const Real gibbs=values.h-temperature*values.entropy;
    if(!gasFinite(gibbs))gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,ugkp::diagnostics::Code::NonFinite,species,gibbs);
    return gasFinite(gibbs)?gibbs:gasThermoDetail::invalidValue<Real>();
}

// Legacy Refinement/Complete template switches remain source-compatible only;
// they never select an alternate evaluator or change result acceptance.
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
    if(!speciesMass)return ThermoStatus::InvalidComposition;
    Real localEnergy=0,localCapacity=0;
    for(int k=0;k<Ns;++k) {
        if(!gasFinite(speciesMass[k]))return gasThermoDetail::failure(view.diagnostics,ThermoStatus::InvalidComposition,ugkp::diagnostics::Code::NonFinite,k,speciesMass[k]);
        gasThermoDetail::SpeciesValues<Real> values;
        const auto status=gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,true>(k,temperature,view,values);
        if(status!=ThermoStatus::Success)return status;
        localEnergy+=speciesMass[k]*values.e;localCapacity+=speciesMass[k]*values.cv;
    }
    if(!gasFinite(localEnergy)||!gasFinite(localCapacity))return gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,ugkp::diagnostics::Code::NonFinite,-1,!gasFinite(localEnergy)?localEnergy:localCapacity);
    energy=localEnergy;heatCapacity=localCapacity;if(evaluated)*evaluated=true;
    return ThermoStatus::Success;
}



template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureInternalEnergy
(
    const Real* speciesMass,Real temperature,const SpeciesThermoView<TableReal,Ns>& view,
    Real& energy,bool* evaluated=nullptr
) noexcept
{
    Real capacity;
    return evaluateMixtureEnergy(speciesMass,temperature,view,energy,capacity,evaluated);
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
    Real local=0;
    for(int k=0;k<Ns;++k){
        if(!gasFinite(speciesMass[k]))return gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,ugkp::diagnostics::Code::NonFinite,k,speciesMass[k]);
        gasThermoDetail::SpeciesValues<Real> values;
        const auto status=gasThermoDetail::evaluateSpecies(k,temperature,view,values);
        if(status!=ThermoStatus::Success)return status;
        local+=speciesMass[k]*values.cv;
    }
    if(!gasFinite(local))return gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,ugkp::diagnostics::Code::NonFinite,-1,local);
    heatCapacity=local;if(evaluated)*evaluated=true;return ThermoStatus::Success;
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
    using namespace gasThermoDetail;
    using namespace ugkp::diagnostics;
    if(!speciesMass || !view.species)return invalidValue<Real>();
    Real totalMass=0,weightedConstant=0;
    for(int k=0;k<Ns;++k) {
        const Real mass=view.species[k].molarMass;
        if(!gasFinite(mass)||mass==Real(0)||!gasFinite(speciesMass[k])) {
            failure(view.diagnostics,ThermoStatus::NonFiniteInput,mass==Real(0)?Code::ZeroDenominator:Code::NonFinite,k,(!gasFinite(mass)||mass==Real(0))?mass:speciesMass[k],(!gasFinite(mass)||mass==Real(0))?Quantity::Denominator:Quantity::Unspecified);
            return invalidValue<Real>();
        }
        totalMass+=speciesMass[k];weightedConstant+=speciesMass[k]*universalGasConstant<Real>()/mass;
    }
    if(totalMass==Real(0)) {
        failure(view.diagnostics,ThermoStatus::InvalidComposition,Code::ZeroDenominator,-1,totalMass,Quantity::Denominator);
        return invalidValue<Real>();
    }
    const Real result=weightedConstant/totalMass;
    if(!gasFinite(result))failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,result);
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
    using namespace gasThermoDetail;
    using namespace ugkp::diagnostics;
    if(!massFractions)return ThermoStatus::InvalidComposition;
    if(!view.species)return ThermoStatus::InvalidModel;
    if(!gasFinite(pressure)||!gasFinite(temperature))return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,!gasFinite(temperature)?temperature:pressure);
    if(temperature<Real(0))return failure(view.diagnostics,ThermoStatus::TemperatureOutOfRange,Code::NegativeTemperature,-1,temperature,Quantity::Temperature);
    Real local[Ns],total=0;
    for(int k=0;k<Ns;++k) {
        const Real y=massFractions[k];
        if(!gasFinite(y))return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,k,y,Quantity::MassFraction);
        local[k]=maxValue(Real(0),minValue(Real(1),y));total+=local[k];
        if(local[k]!=y)recordWarning(view.diagnostics,makeRecord(Code::FiniteYClip,2304,0,0,EntityKind::Species,k,k,y,0,local[k],Quantity::MassFraction));
    }
    if(absValue(total-Real(1))>compositionTolerance)recordWarning(view.diagnostics,makeRecord(Code::ConservationResidual,10004,0,0,EntityKind::None,-1,-1,total,1,total,Quantity::Conservation));
    const Real gasConstant=mixtureGasConstant(local,view);
    const Real denominator=gasConstant*temperature;
    if(!gasFinite(denominator)||denominator==Real(0))return failure(view.diagnostics,ThermoStatus::NonFiniteInput,denominator==Real(0)?Code::ZeroDenominator:Code::NonFinite,-1,denominator,Quantity::Denominator);
    const Real result=pressure/denominator;
    if(!gasFinite(result))return failure(view.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,result,Quantity::Density);
    density=result;return ThermoStatus::Success;
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
    if(evaluated)*evaluated=false;
    if(status!=ThermoStatus::Success) return status;
    if(!gasFinite(localEnergy) || !gasFinite(localCapacity)) {
        return ThermoStatus::NonFiniteInput;
    }
    energy=localEnergy;capacity=localCapacity;if(evaluated)*evaluated=true;return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal,bool Refinement> struct MixtureEnergyEvaluator;



template<bool Refinement=false,class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus invertCaloricEnergy
(
    const Evaluator& evaluator,Real internalEnergy,Real lower,Real upper,
    const ThermoInversionControls<Real>& controls,Real& temperature,bool* evaluatedTemperature=nullptr,
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr
) noexcept
{
    using namespace gasThermoDetail;
    using namespace ugkp::diagnostics;
    if(evaluatedTemperature)*evaluatedTemperature=false;
    if(!gasFinite(internalEnergy))return failure(diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,internalEnergy,Quantity::InternalEnergy);
    Real lo=maxValue(lower,controls.minTemperature);
    Real hi=controls.maxTemperature>Real(0)?minValue(upper,controls.maxTemperature):upper;
    Real elo,clo,ehi,chi;
    auto status=evaluateCaloricEnergy<false>(evaluator,lo,elo,clo);
    if(status!=ThermoStatus::Success)return status;
    status=evaluateCaloricEnergy<false>(evaluator,hi,ehi,chi);
    if(status!=ThermoStatus::Success)return status;
    if(internalEnergy==elo){temperature=lo;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success;}
    if(internalEnergy==ehi){temperature=hi;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success;}
    const bool bracket=lo<hi && ((elo<internalEnergy && ehi>internalEnergy)||(elo>internalEnergy && ehi<internalEnergy));
    Real trial=gasFinite(temperature) && temperature>lo && temperature<hi?temperature:lo+Real(.5)*(hi-lo);
    const Real energyTolerance=controls.absoluteEnergyTolerance+controls.relativeEnergyTolerance*absValue(internalEnergy);
    const int iterations=controls.maximumIterations>0?controls.maximumIterations:1;
    Real residual=0;
    for(int iteration=0;iteration<iterations;++iteration) {
        Real energy,capacity;
        status=evaluateCaloricEnergy<false>(evaluator,trial,energy,capacity);
        if(status!=ThermoStatus::Success)return status;
        residual=energy-internalEnergy;
        if(!gasFinite(residual)) {
            if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,30600,0,0,EntityKind::None,-1,-1,residual,0,0,Quantity::InternalEnergy));
            return ThermoStatus::NonFiniteInput;
        }
        if(capacity==Real(0))return failure(diagnostics,ThermoStatus::NonPositiveHeatCapacity,Code::ZeroDenominator,-1,capacity,Quantity::Denominator);
        const Real correction=residual/capacity;
        if(!gasFinite(correction)) {
            if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,30601,0,0,EntityKind::None,-1,-1,correction,0,0,Quantity::Temperature));
            return ThermoStatus::NonFiniteInput;
        }
        const Real tolerance=controls.absoluteTemperatureTolerance+controls.relativeTemperatureTolerance*absValue(trial);
        if(absValue(residual)<=energyTolerance && absValue(correction)<=tolerance){
            temperature=trial;if(evaluatedTemperature)*evaluatedTemperature=true;return ThermoStatus::Success;
        }
        if(iteration+1==iterations)break;
        if(bracket){if((residual<Real(0))==(elo<internalEnergy)){lo=trial;elo=energy;}else{hi=trial;ehi=energy;}}
        Real next=trial-correction;
        // Safeguarded mathematical root iteration, not a failed physical-step retry.
        if(bracket && (!gasFinite(next)||!(next>lo && next<hi)))next=lo+Real(.5)*(hi-lo);
        if(!gasFinite(next))return failure(diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,next,Quantity::Temperature);
        if(next==trial)break;
        trial=next;
    }
    temperature=trial;if(evaluatedTemperature)*evaluatedTemperature=true;
    recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,2303,0,0,EntityKind::None,-1,-1,residual,energyTolerance,residual,Quantity::Accuracy));
    return ThermoStatus::Success;
}

template<class Real,int Ns,class TableReal=Real,bool Refinement=false>
struct MixtureEnergyEvaluator
{
    const Real* speciesMass;
    SpeciesThermoView<TableReal,Ns> thermo;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    { return evaluateMixtureEnergy<Refinement,Real,Ns,TableReal,false>(speciesMass,T,thermo,energy,capacity); }
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
    return invertCaloricEnergy<Refinement>(evaluator,internalEnergy,lo,hi,controls,temperature,evaluatedTemperature,view.diagnostics);
}

} // namespace ugkwp
#endif
