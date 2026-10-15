#pragma once
#include <type_traits>
#include <cmath>
#include <utility>
#include "GasModel.H"

namespace ugkwp
{
// The pointed-to scalar is authoritative. Never infer energy precision from rho
// or pass these values through a Real-only clamp/finite helper.
template<class State>
using GasEnergyScalar = typename std::remove_cv<typename std::remove_reference
<decltype(std::declval<State>().rhoE[0])>::type>::type;

template<class Energy>
UGKWP_GAS_HD inline bool gasEnergyFinite(Energy value) noexcept
{
    return std::isfinite(value);
}



template<class Energy>
UGKWP_GAS_HD inline bool gasEnergyFallbackFinite(Energy value) noexcept
{ return value == value && value < 1.0e300 && value > -1.0e300; }

template<class Energy>
UGKWP_GAS_HD inline Energy gasEnergyFiniteOr(Energy value, Energy fallback) noexcept
{ return gasEnergyFallbackFinite(value) ? value : fallback; }

template<class Energy>
UGKWP_GAS_HD inline Energy gasEnergyClampMin(Energy value, Energy minimum) noexcept
{ return value < minimum ? minimum : value; }
// Existing fixed-grid legacy EOS caloric conventions. Share these with heat
// acceptance so its admissible budget is exactly the recovery budget.
template<class Energy>
UGKWP_GAS_HD inline Energy gasLegacyCaloricDenominator
(Energy gamma, Energy small) noexcept
{ return gasEnergyClampMin(gamma-Energy(1),small); }

template<class Energy>
UGKWP_GAS_HD inline Energy gasLegacyThermalMinimumInternalEnergy
(Energy density, Energy gasConstant, Energy temperatureMin,
 Energy gamma, Energy small) noexcept
{
    return density*gasConstant*temperatureMin/gasLegacyCaloricDenominator(gamma,small);
}

template<class Energy>
UGKWP_GAS_HD inline Energy gasLegacyMinimumInternalEnergy
(Energy rhoMin, Energy density, Energy gasConstant, Energy temperatureMin,
 Energy gamma, Energy small) noexcept
{
    return gasEnergyClampMin(rhoMin, gasLegacyThermalMinimumInternalEnergy
        (density,gasConstant,temperatureMin,gamma,small));
}

template<class Energy>
UGKWP_GAS_HD inline Energy gasLegacyKineticEnergy
(Energy density, Energy ux, Energy uy, Energy uz) noexcept
{ return Energy(0.5)*density*(ux*ux+uy*uy+uz*uz); }

}
