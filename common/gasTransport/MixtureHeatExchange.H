#ifndef UGKWP_MIXTURE_HEAT_EXCHANGE_H
#define UGKWP_MIXTURE_HEAT_EXCHANGE_H
#include "MixtureThermo.H"

namespace ugkwp
{
// Density form of V*(E(T)-Eold) + A*(T-Tref) = sum_i w_i*(Tp_i-Tref).
// Centering removes formation-energy offsets from the convergence scale. The
// parcel temperatures enter the right hand side unchanged; only gas property
// evaluations are restricted to the gas table's valid interval.
template<class Real,int Ns,class TableReal>
struct MixtureHeatExchangeEvaluator
{
    MixtureEnergyEvaluator<Real,Ns,TableReal,false> mixture;
    Real oldEnergy,linearCapacity,referenceTemperature,lower,upper;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    {
        if(!gasFinite(T))return ThermoStatus::NonFiniteInput;
        if(T<lower || T>upper)return ThermoStatus::TemperatureOutOfRange;
        const auto status=mixture(T,energy,capacity);
        if(status!=ThermoStatus::Success)return status;
        energy=(energy-oldEnergy)+linearCapacity*(T-referenceTemperature);
        capacity+=linearCapacity;
        return ThermoStatus::Success;
    }
};

template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus invertMixtureHeatExchange
(
    const Real* speciesMass,Real oldEnergy,Real linearCapacity,Real heatTarget,
    const SpeciesThermoView<TableReal,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!view.species || !speciesMass)return ThermoStatus::InvalidModel;
    Real lo,hi;mixtureTemperatureBounds(view,controls,lo,hi);
    if(!(lo<hi))return ThermoStatus::InvalidControls;
    const MixtureHeatExchangeEvaluator<Real,Ns,TableReal> evaluator
        {{speciesMass,view},oldEnergy,linearCapacity,temperature,lo,hi};
    // Subtracting represented caloric energies has this roundoff floor even
    // when the requested physical heat transfer is much smaller than Eold.
    auto solveControls=controls;
    const Real energyRoundoff=Real(4)*std::numeric_limits<Real>::epsilon()
        *(absValue(oldEnergy)+absValue(heatTarget));
    solveControls.absoluteEnergyTolerance+=energyRoundoff;
    Real candidate=temperature;
    const auto status=invertCaloricEnergy(evaluator,heatTarget,lo,hi,solveControls,
                                         candidate,nullptr,view.diagnostics);
    if(status!=ThermoStatus::Success)return status;
    Real energy,capacity;
    const auto evaluated=evaluateCaloricEnergy(evaluator,candidate,energy,capacity);
    if(evaluated!=ThermoStatus::Success)return evaluated;
    const Real residual=energy-heatTarget;
    const Real energyTolerance=solveControls.absoluteEnergyTolerance
        +solveControls.relativeEnergyTolerance*absValue(heatTarget);
    const Real temperatureTolerance=solveControls.absoluteTemperatureTolerance
        +solveControls.relativeTemperatureTolerance*absValue(candidate);
    // invertCaloricEnergy also supports warning-only accuracy diagnostics.
    // A paired exchange must not report success solely from a closed budget:
    // its constitutive equation must have converged as well.
    if(capacity==Real(0))return ThermoStatus::NonPositiveHeatCapacity;
    if(absValue(residual)>energyTolerance
       || absValue(residual/capacity)>temperatureTolerance+energyRoundoff/absValue(capacity))
        return ThermoStatus::IterationLimit;
    temperature=candidate;
    return ThermoStatus::Success;
}
} // namespace ugkwp
#endif
